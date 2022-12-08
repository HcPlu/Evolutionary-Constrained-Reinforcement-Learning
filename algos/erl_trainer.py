
import numpy as np, os, time, random, torch, sys
from algos.neuroevolution import SSNE
from core import utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
from core.buffer import Buffer, constraintBuffer
import torch
import tianshou as ts

device_num = torch.cuda.device_count()
device_list = ["cuda:%d"%i for i in range(device_num)]

class ERL_Trainer:

	def __init__(self, args, sac_policy, env_constructor, model_constructor):

		self.args = args
		self.manager = Manager()
		self.device = args.device



		#RCPO: lambda
		#todo: parameters
		self.rcpo_lambda = args.rcpo_lambda
		self.rcpo_lr = args.rcpo_lr
		self.rcpo_alpha = args.rcpo_alpha

		#SRECRL
		self.constraint_buffer = constraintBuffer(100)
		if self.rcpo_lr > 0:
			self.population_lambda = np.random.random(args.pop_size)
		else:
			self.population_lambda = np.ones(args.pop_size)*self.rcpo_lambda
		self.lambda_batch_num = 32


		#Evolution
		self.evolver = SSNE(self.args)

		#Initialize population
		self.population = self.manager.list()
		for id in range(args.pop_size):
			self.population.append(model_constructor.build_actor(device = device_list[id%device_num]))

		#Save best policy
		self.best_policy = model_constructor.build_actor(device = device_list[0%device_num])

		self.learner = sac_policy

		#Replay Buffer
		self.replay_buffer = ts.data.ReplayBuffer(args.buffer_size)

		#Initialize Rollout Bucket
		self.rollout_bucket = self.manager.list()
		for id in range(args.rollout_size):
			self.rollout_bucket.append(model_constructor.build_actor(device = device_list[id%device_num]))

		############## MULTIPROCESSING TOOLS ###################
		#Evolutionary population Rollout workers
		self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]

		self.evo_workers = [Process(target=rollout_worker, args=(id, 'evo', self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], args.rollout_size > 0, self.population, env_constructor)) for id in range(args.pop_size)]
		for worker in self.evo_workers: worker.start()
		self.evo_flag = [True for _ in range(args.pop_size)]

		#Learner rollout workers

		self.task_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.result_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.workers = [Process(target=rollout_worker, args=(id, 'pg', self.task_pipes[id][1], self.result_pipes[id][0], True, self.rollout_bucket, env_constructor)) for id in range(args.rollout_size)]
		for worker in self.workers: worker.start()
		self.roll_flag = [True for _ in range(args.rollout_size)]

		#Test bucket
		self.test_bucket = self.manager.list()
		for id in range(args.num_test):
			self.test_bucket.append(model_constructor.build_actor(device = device_list[id%device_num]))

		# Test workers

		self.test_task_pipes = [Pipe() for _ in range(args.num_test)]
		self.test_result_pipes = [Pipe() for _ in range(args.num_test)]
		self.test_workers = [Process(target=rollout_worker, args=(id, 'test', self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, self.test_bucket, env_constructor)) for id in range(args.num_test)]
		for worker in self.test_workers: worker.start()
		self.test_flag = False

		#Trackers
		self.best_score = -float('inf'); self.gen_frames = 0; self.total_frames = 0; self.test_score = None; self.test_std = None

	def update_rcpo_lambda(self, Jcs):
		Jcs = np.array(Jcs)
		dlambda = Jcs - np.ones(Jcs.shape)*self.rcpo_alpha
		self.rcpo_lambda = max(self.rcpo_lambda+np.mean(self.rcpo_lr*dlambda),0)

	def forward_generation(self, gen, tracker):

		gen_max = -float('inf')

		#Start Evolution rollouts
		if self.args.pop_size > 1:
			for id, actor in enumerate(self.population):
				pop_lambda = self.population_lambda[id]
				self.evo_task_pipes[id][0].send([id,pop_lambda])

		#Sync all learners actor to cpu (rollout) actor and start their rollout
		for rollout_id in range(len(self.rollout_bucket)):
			utils.hard_update(self.rollout_bucket[rollout_id], self.learner.actor)
			self.task_pipes[rollout_id][0].send([rollout_id,self.rcpo_lambda])


		#Start Test rollouts
		if gen % self.args.test_frequency == 0:
			self.test_flag = True
			self.learner.eval()
			for id,pipe in enumerate(self.test_task_pipes):
				pipe[0].send([id,self.rcpo_lambda])


		############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		if self.replay_buffer.__len__() > self.args.learning_start: ###BURN IN PERIOD
			# self.learner.critic2.to(device=self.device)
			# self.learner.critic1.to(device=self.device)
			self.learner.train()
			for _ in range(int(self.gen_frames * self.args.gradperstep)):
				losses = self.learner.update(self.args.batch_size, self.replay_buffer)
				# s, ns, a, r,c, done = self.replay_buffer.sample(self.args.batch_size)
				# #todo: constraint (SAC+RCPO)
				# self.learner.update_parameters(s, ns, a, r, done)
			self.gen_frames = 0


		########## JOIN ROLLOUTS FOR EVO POPULATION ############
		all_reward_fitness = [];all_constraint_fitness = []; all_eplens = []
		if self.args.pop_size > 1:
			for i in range(self.args.pop_size):
				_, reward_fitness, constraint_fitness, frames, trajectory = self.evo_result_pipes[i][1].recv()
				all_reward_fitness.append(reward_fitness);all_constraint_fitness.append(constraint_fitness);all_eplens.append(frames)
				self.gen_frames+= frames; self.total_frames += frames
				# self.replay_buffer.add(trajectory)
				utils.add_steps(self.replay_buffer,trajectory)
				self.best_score = max(self.best_score, reward_fitness)
				gen_max = max(gen_max, reward_fitness)

		self.constraint_buffer.add(all_constraint_fitness)

		########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		rollout_reward_fitness = [];rollout_constraint_fitness = []; rollout_eplens = []
		if self.args.rollout_size > 0:
			for i in range(self.args.rollout_size):
				_, reward_fitness, constraint_fitness, pg_frames, trajectory = self.result_pipes[i][1].recv()
				# self.replay_buffer.add(trajectory)
				utils.add_steps(self.replay_buffer, trajectory)
				self.gen_frames += pg_frames; self.total_frames += pg_frames
				self.best_score = max(self.best_score, reward_fitness)
				gen_max = max(gen_max, reward_fitness)
				rollout_reward_fitness.append(reward_fitness); rollout_constraint_fitness.append(constraint_fitness);rollout_eplens.append(pg_frames)

		self.constraint_buffer.add(rollout_constraint_fitness)
		######################### END OF PARALLEL ROLLOUTS ################

		############ FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST #############
		if self.args.pop_size > 1:
			champ_index = all_reward_fitness.index(max(all_reward_fitness))
			for id in range(len(self.test_bucket)):
				utils.hard_update(self.test_bucket[id], self.population[champ_index])
			if max(all_reward_fitness) >= self.best_score:
				self.best_score = max(all_reward_fitness)
				utils.hard_update(self.best_policy, self.population[champ_index])
				torch.save(self.population[champ_index].state_dict(), self.args.aux_folder + '_best'+self.args.savetag)
				print("Best policy saved with score", '%.2f'%max(all_reward_fitness))

		else: #If there is no population, champion is just the actor from policy gradient learner
			for id in range(len(self.test_bucket)):
				utils.hard_update(self.test_bucket[id], self.rollout_bucket[0])
				# utils.hard_update(self.test_bucket[id], self.population[champ_index])

		# RCPO: update lambda
		if self.replay_buffer.__len__() > self.args.learning_start:
			if self.rcpo_lr>0:
				self.update_rcpo_lambda(all_constraint_fitness+rollout_constraint_fitness)
		self.args.writer.add_scalar('lambda', self.rcpo_lambda, gen)
			# print(batch)


		###### TEST SCORE ######
		if self.test_flag:
			self.test_flag = False
			test_scores = [];test_constraints = []
			for pipe in self.test_result_pipes: #Collect all results
				_, reward_fitness, constraint_fitness, _, _ = pipe[1].recv()
				self.best_score = max(self.best_score, reward_fitness)
				gen_max = max(gen_max, reward_fitness)
				test_scores.append(reward_fitness)
				test_constraints.append(constraint_fitness)
			test_scores = np.array(test_scores)
			test_constraints = np.array(test_constraints)
			test_reward_mean = np.mean(test_scores); test_reward_std = (np.std(test_scores))
			test_constraint_mean = np.mean(test_constraints);test_constraint_std = (np.std(test_constraints))
			tracker.update([test_reward_mean,test_reward_std,test_constraint_mean,test_constraint_std], self.total_frames)

		else:
			test_reward_mean, test_reward_std = None, None


		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size > 1:
			leaner_replaces = self.evolver.epoch(gen, self.population, all_reward_fitness,all_constraint_fitness, [self.rollout_bucket[0]],self.rcpo_alpha)
		# update lambda population
			if self.rcpo_lr>0:
				for i in leaner_replaces:
					lambda_batch = self.constraint_buffer.sample(self.lambda_batch_num)
					Jcs = np.array(lambda_batch)
					dlambda = Jcs - np.ones(Jcs.shape) * self.rcpo_alpha
					self.population_lambda[i] = max(self.population_lambda[i] + np.mean(self.rcpo_lr * dlambda), 0)

		self.args.writer.add_scalar('pop_lambda', np.mean(self.population_lambda), gen)
		#Compute the champion's eplen
		champ_len = all_eplens[all_reward_fitness.index(max(all_reward_fitness))] if self.args.pop_size > 1 else rollout_eplens[rollout_reward_fitness.index(max(rollout_reward_fitness))]


		return gen_max, champ_len, all_eplens, test_reward_mean, test_reward_std,test_constraint_mean, test_constraint_std, rollout_reward_fitness, rollout_eplens


	def train(self, frame_limit):
		# Define Tracker class to track scores
		test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag], '.csv')  # Tracker class to log progress
		time_start = time.time()
		print("STARTING")
		for gen in range(1, 1000000000):  # Infinite generations

			# Train one iteration
			max_fitness, champ_len, all_eplens, test_reward_mean, test_reward_std, test_constraint_mean, test_constraint_std, rollout_fitness, rollout_eplens = self.forward_generation(gen, test_tracker)
			if test_reward_mean:
				self.args.writer.add_scalar('test_score', test_reward_mean, gen)
				self.args.writer.add_scalar('test_constraint', test_constraint_mean, gen)

			print('Gen/Frames:', gen,'/',self.total_frames,
				  ' Gen_max_score:', '%.2f'%max_fitness,
				  ' Champ_len', '%.2f'%champ_len, ' Test_score u/std', utils.pprint(test_reward_mean), utils.pprint(test_reward_std),utils.pprint(test_constraint_mean), utils.pprint(test_constraint_std),
				  ' Rollout_u/std:', utils.pprint(np.mean(np.array(rollout_fitness))), utils.pprint(np.std(np.array(rollout_fitness))),
				  ' Rollout_mean_eplen:', utils.pprint(sum(rollout_eplens)/len(rollout_eplens)) if rollout_eplens else None)

			if gen % 5 == 0:
				print('Best_score_ever:''/','%.2f'%self.best_score, ' FPS:','%.2f'%(self.total_frames/(time.time()-time_start)), 'savetag', self.args.savetag)
				print()

			if self.total_frames > frame_limit:
				torch.save(self.learner.actor.cpu().state_dict(), self.args.aux_folder + '_learner_final'+self.args.savetag)
				break

		###Kill all processes
		try:
			for p in self.task_pipes: p[0].send(['TERMINATE',-1])
			for p in self.test_task_pipes: p[0].send(['TERMINATE',-1])
			for p in self.evo_task_pipes: p[0].send(['TERMINATE',-1])
		except:
			None




