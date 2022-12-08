import os
from torch.utils.tensorboard import SummaryWriter
import json
class Parameters:
    def __init__(self, parser):
        """Parameter class stores all parameters for policy gradient

        Parameters:
            None

        Returns:
            None
        """

        #Env args
        print(vars(parser.parse_args()))
        self.env_name = vars(parser.parse_args())['env']
        self.frameskip = vars(parser.parse_args())['frameskip']
        self.state_shape = None
        self.action_sahpe = None
        self.max_action = None
        self.auto_alpha = vars(parser.parse_args())["auto_alpha"]
        self.total_steps = int(vars(parser.parse_args())['total_steps'] * 1000000)
        self.gradperstep = vars(parser.parse_args())['gradperstep']
        self.savetag = vars(parser.parse_args())['savetag']
        self.seed = vars(parser.parse_args())['seed']
        self.batch_size = vars(parser.parse_args())['batchsize']
        self.rollout_size = vars(parser.parse_args())['rollsize']
        self.n_step = vars(parser.parse_args())["n_step"]
        self.load_path = vars(parser.parse_args())["load_path"]
        self.fitness_rank = vars(parser.parse_args())["fitness_rank"]

        self.rcpo_lambda = vars(parser.parse_args())["rcpo_lambda"]
        self.rcpo_lr = vars(parser.parse_args())["rcpo_lr"]
        self.rcpo_alpha = vars(parser.parse_args())["rcpo_alpha"]

        self.hidden_sizes = vars(parser.parse_args())['hidden_sizes']
        self.critic_lr = vars(parser.parse_args())['critic_lr']
        self.actor_lr = vars(parser.parse_args())['actor_lr']
        self.tau = vars(parser.parse_args())['tau']
        self.gamma = vars(parser.parse_args())['gamma']
        self.reward_scaling = vars(parser.parse_args())['reward_scale']
        self.buffer_size = int(vars(parser.parse_args())['buffer'] * 1000000)
        self.learning_start = vars(parser.parse_args())['learning_start']
        self.device = vars(parser.parse_args())["device"]
        self.pop_size = vars(parser.parse_args())['popsize']
        self.num_test = vars(parser.parse_args())['num_test']
        self.test_frequency = 1
        self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution

        #Non-Args Params
        self.elite_fraction = 0.2
        self.crossover_prob = 0.15
        self.mutation_prob = 0.90
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform


        self.alpha = vars(parser.parse_args())['alpha']
        self.target_update_interval = 1
        self.alpha_lr = 1e-3

        #save tage
        self.savetag += str(self.env_name)
        self.savetag += '_seed' + str(self.seed)
        self.savetag += '_roll' + str(self.rollout_size)
        self.savetag += '_pop' + str(self.pop_size)
        self.savetag += '_alpha' + str(self.alpha)


        #Save Results
        result_path = 'cerl_tianshou'
        result_path +='_lr'+str(self.rcpo_lr)
        result_path += "_"+str(self.rcpo_lambda)
        if self.rcpo_alpha!=0.4:
            result_path += "_" + str(self.rcpo_alpha)
        if self.fitness_rank:
            result_path += "_nosr"

        self.savefolder = result_path+'/Plots/'+self.savetag+"/"
        if not os.path.exists(self.savefolder): os.makedirs(self.savefolder)
        self.aux_folder = result_path+'/Auxiliary/'+self.savetag+"/"
        if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)
        with open(result_path+"/params.json","w") as f:
            json.dump(vars(parser.parse_args()),f)
        self.writer = SummaryWriter(log_dir=result_path+'/tensorboard/' + self.savetag)




