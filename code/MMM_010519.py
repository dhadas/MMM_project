
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import warnings
import json

num_of_topics = 12


class Theta:

    def __init__(self):

        """
        Parameters
         ----------
         e: np.ndarray.
             A matrix where e i,j is the probability to choose a word j  with signature i.

         pie: np.ndarray.
             An array of where pie i is the probability to choose signature i.

         Emeans: np.ndarray.
             A matrix of means, where Emeans i,j is the mean of probability that word j from theta came from signature i.

        """

        self.e = np.zeros((num_of_topics, 96))
        self.pie = np.zeros(num_of_topics)
        self.Emeans = np.zeros((num_of_topics, 96))



    def copy(self):

        new = Theta()
        new.pie = self.pie.copy()
        new.e = self.e.copy()
        new.Emeans = self.Emeans.copy()
        return new

    # def build_random_theta(self) -> Theta:
    def build_random_theta(self):
        """

            Builds a Theta by randomly selecting pie and e values.
            v1.0 - values are randomized using the normal distribution
                and then normalized again to sum up to 1 for each signature.

            Returns
            -------
            self: Theta.
                A Theta object with randomized pie and e values.
        """

        #Generate random pi values and then normalize it
        self.pie = np.random.rand(num_of_topics)
        s = sum(self.pie)
        self.pie /= s

        #Generate random eij matrix where (i,j) is P(word is j | signiture is i)
        self.e = np.zeros((num_of_topics, 96))
        for i in range (num_of_topics):
            #generate random vector for each sig
            self.e[i] = np.random.rand(96)
            #normalize probabilities
            s = sum(self.e[i])
            self.e[i] /= s

        #get log values of the randomized values
        self.e = np.log(self.e)
        self.pie = np.log(self.pie)
        return self

    def fill_Emeans(self, B):
        """
            Updates the Emeans matrix of theta by computing the probability that word j from theta came from signature i.

            Parameters
            ----------
            B: np.array.
                A numpy array where B i is the number of times word i was given as input.

            Returns
            -------
            int
                1 if successful.

        """


        for i in range(self.e.shape[1]):
            #for each word
            if B[i] == 0: #if did not appear, set appearences to zero
                self.Emeans[:, i] = 0
            else: #Else calculate Bj * P(y_t = i | x_t = j)
                # self.Emeans[:, i] = (np.add(self.e[:, i], self.pie)) - logsumexp(self.e.T[i] + self.pie) + np.log(B[i])
                self.Emeans[:, i] = (np.add(self.e[:, i], self.pie)) - logsumexp(self.e.T[i] + self.pie) + np.log(B[i])
        return 1


    def theta_t(self,E_res):
        """
            Builds a new Theta object by computing its pie and e values, using the current theta.

            Returns
            -------
            Theta
                A theta with pie and e values. The E matrices are None.
        """

        theta_new = Theta()
        theta_new.e = self.e.copy()
        # print(self.Emeans, self.Emeans.sum(axis=0))
        # for i in range(96):
        #     theta_new.e[:,i] = np.subtract(E_res[:,i], logsumexp(E_res,axis=1))
        # theta_new.e[np.isnan(theta_new.e)] = 0
        e_summed = np.zeros(num_of_topics)
        for i in range(E_res.shape[0]):
            # print(E_res)
            e_summed[i] = logsumexp(E_res[i])
            # print(e_summed[i])
        # print(logsumexp(self.Emeans))
        np.subtract(e_summed, logsumexp(self.Emeans), theta_new.pie)
        # print(theta_new.pie)


        return theta_new
    @staticmethod
    def log_likelihood(X,array):

        """
            Computes the log-likelihood of X ,vector of chosen words, given theta.

            Parameters
            ----------
            X: np.array.
                A numpy array of words.

            Returns
            -------
            float.
                The log likelihood of X given theta
        """
        res_array = np.zeros((560))
        for j in range(560):
            arr = np.zeros(96)
            for i in range(array[j].e.shape[1]):
                # print(self.e.T[i])
                arr[i] = logsumexp(array[j].e.T[i] + array[j].pie) + np.log(X[X.columns[j]].loc[i])
            res_array[j] = logsumexp(arr)
        return logsumexp(res_array)

    @staticmethod
    def log_likelihood2(X, array) -> np.array:
        res_array = np.zeros((560))
        for j in range(560):
            curr_theta = array[j]
            temp = curr_theta.e.T + curr_theta.pie
            joint_prob = logsumexp(temp, axis=1)
            curr_LL = np.dot(joint_prob, X.iloc[:,j])
            # print(curr_LL)

            res_array[j] = curr_LL

        return logsumexp(res_array)


    def check_convergence(self, X, threshold,arr_new,arr_old):
        """
            Computes the log-likelihood of X ,vector of chosen words, given theta

            Parameters
            ----------
            theta_old: Theta.
                The theta to which we check convergence.

            X: np.array.
                A numpy array of words.

            threshold: float.
                The threshold of log likelihood for the model.

            Returns
            -------
            int.
                1 if the threshold is met, 0 otherwise.
        """


        if (Theta.log_likelihood2(X,arr_new) - Theta.log_likelihood2(X,arr_old) < threshold):
            return 1
        else:
            return 0


def set_num_of_topics(n: int):
    global num_of_topics
    num_of_topics = n
    return


def generate_LL_df(model_name: str, k: int, E_mat: pd.DataFrame, chrom_lst : list, n: int = 10):
    # Create a dictionary of LL lists for each chromosome, later to be converted to a DataFrame
    ll_dict = {}
    model_name = model_name  # Change to whaterver model you are testing
    set_num_of_topics(k)   # Change to desired number of topics
    # print("Num of topics is : " + str(num_of_topics))

    # Iterate Through chromosomes, preform cross validation for each chromosomes.
    for chrom_id in chrom_lst:
        # held out chromosme will later use for testing
        test = pd.read_csv(f"data/cross_val_datasets/chrom{chrom_id}.csv").drop("Unnamed: 0", axis=1)
        # dataset with a single chromosome out
        training_set = pd.read_csv(f"data/cross_val_datasets/full_data_chrom{chrom_id}out.csv").drop("Unnamed: 0", axis=1)

        print(f"Now training MMM model, on full dataset with chrom #{chrom_id} out.\n")
        ll_dict[f'{model_name}_{num_of_topics}_{chrom_id}'] = []

        # For each chromosome generate 10 results to overcome random data
        for i in range(n):
            print(f"Iteration: #{i} : ", end="")
            Model = fit(training_set, 0.001, 50, E_mat)  # Change input matrix to whichever matrix you want
            if (Model == 1):  # if model didnot converge, 1 is appended instead of a negative likelihood score
                print("Did not converrge, appending 1\n")
                ll_dict[f'{model_name}_{num_of_topics}_{chrom_id}'].append(1.0)
            else:
                # Else model converged, check LL and append it.
                print("Model converged, appending LL score: ", end="")
                ll_dict[f'{model_name}_{num_of_topics}_{chrom_id}'].append(Theta.log_likelihood2(test, Model))
                print(ll_dict[f'{model_name}_{num_of_topics}_{chrom_id}'][i])

            print(f"Finished Iteration: {i}\n")

    return pd.DataFrame(ll_dict)


def fix_count(X, len = 96):

    """
    Takes as input a vector with indices and count how many time each one of them was seen.
    Limited to 96 for now.

    Parameters
    ----------
    X: np.array
        A numpy array of words.

    Returns
    -------
    B: np.array.
        A numpy array where B i is the number of times word i was given as input.

    """

    #Get unique values and the number of their appearences in X
    vals, counts = np.unique(X, return_counts=True)
    B = np.zeros(len)
    counter = 0
    for i in vals:
        B[i] = counts[counter]
        counter +=1

    return B



def fit(X, threshold, max_iterations, fixed_e : pd.DataFrame = None, init_pie : pd.DataFrame = None):
    """
        fit the model to the data X until threshold is met or until max_iterations.

        Parameters
        ----------
        threshold: float.
            The threshold of log likelihood for the model

        max_iterations: int.
            Number of maximal EM iterations to perform.

        Returns
        -------
        Theta
            The theta that gives the best score for a given X words vector.

        int flag
            1 if the threshold is met, 0 in case num of EM iterations reached max_iterations
    """
    '''
    if (X == []):
        raise ValueError('Invalid input, word vector X is empty')
    if (max_iterations < 1):
        raise ValueError('Invalid input, max_iterations has to be >= 1, provided 0 or less')
    '''

    #Get occurences matrix B
    # B = fix_count(X) # needs fixing
    #Read intial e values

    ##Needs to be an array - of 560 theta
    arr = []
    #Initiate current theta model

    for i in range(560):
        theta_curr = Theta()
        theta_curr.build_random_theta()
        arr.append(theta_curr)

    # If an initial pi matrix wass provided, assign it by order to each theta model (representing a single person)
    if init_pie is not None:
        print("Initial pi provided!, assigninig values. \n")
        for i in range(560):
            arr[i].pie = np.array(init_pie.iloc[i].values[1:]).astype(np.float64)
            arr[i].pie = np.log(arr[i].pie)




    #Fill theta with randomized values then transform to log of values.

    #theta_curr is now randomized and transformed to log

    #For checking basic model works
    #######
    # theta_curr.pie = np.array([0.23305193248719497, 0.0014132314926197209, 0.20416660764668343, 0.00020190401189555402, 0.20084343287046813, 0.15847818668246588, 0.013389301125281252, 0.0022095205339864228, 0.08674215833197821, 0.0671392537284252, 0.01933872390541033, 0.013025747183590633])
    if( fixed_e is not None):
        for i in range(560):
            arr[i].e = np.log(fixed_e)



    # print(theta_curr.pie)
    # theta_curr.pie = np.log(theta_curr.pie)
    # print(theta_curr.pie)
    # theta_curr.e = np.log(theta_curr.e)
    E_res = np.zeros((num_of_topics,96))

    for i in range(560):
        arr[i].fill_Emeans(X[X.columns[i]])
        np.logaddexp(E_res,arr[i].Emeans,E_res)
    E_res = E_res - np.log(560)
    # print(E_res.shape)
    #######
    # print(E_res)

    # print(theta_curr.pie)
    # print(theta_curr.e)
    iter = 0
    conv = 0
    E_res1 = np.zeros((num_of_topics,96))

    while (conv == 0 and iter < max_iterations):
        # print(theta_curr.e, theta_curr.e.sum())
        # print("\n")
        # Store previous model theta
        # print(theta_curr.pie)
        arr_old = arr.copy()
        # print(arr[0].pie)
        for i in range(560):
            theta_prev = arr[i].copy()
        # maximze
            arr[i] = theta_prev.theta_t(E_res)
        #recalculate E step
            arr[i].fill_Emeans(X[X.columns[i]])
            np.logaddexp(E_res1,arr[i].Emeans,E_res1)
        # print(arr[0].pie)
        E_res1 = E_res1 - np.log(560)
        E_res = E_res1.copy()
        # Check
        conv = theta_curr.check_convergence(X, threshold,arr,arr_old)
        # print(conv)
        # if(iter > 5):
        #     print(theta_prev.log_likelihood(X))
        #
        #     print(theta_curr.log_likelihood(X))
        #     print("\n")
        iter += 1
        # print(iter)

    if(conv == 0):
        print("did not converge!")
        return 1
    else:
        print('Converged after ' + str(iter)+ ' iterations')

    return arr




def transform_chr_to_np(chr_dict):
    lst = []
    # print list(chr_dict.keys())
    for chr in chr_dict:
        # print "Transforming\n"
        # print(chr_dict[chr]['Sequence'])
        lst.extend(chr_dict[chr]['Sequence'])

    return np.array(lst)



# with open('data/ICGC-BRCA.json') as json_file:
#     data = json.load(json_file)
#     # print(data)
#     first = True
#
#     for p in data:
#         if(first):
#             X = transform_chr_to_np(data[p])
#             first = False
#         else:
#             X =  np.block([X, transform_chr_to_np(data[p])])
#             # X = transform_chr_to_np(data[p])


warnings.filterwarnings("ignore", message="divide by zero encountered in log")

X = pd.read_csv("data/010519/lets_do_some_shit.csv",index_col=0)

'''
given_e_mat = pd.read_csv("data/given_E12_matrix.csv", index_col=0).values
print(X)
print X.shape
mmm_560 = fit(X, 0.001, 100, given_e_mat)[0]

MMM_res = (np.exp(mmm_560.pie))
print "Our pi is:\n ", MMM_res

'''

'''

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95])
y = fit(x, 0.0001, 1000)[0]
res = (np.exp(y.pie))
print("Our pi is:\n ", res)
trained_pi =  ([0.01991760467312221, 7.702496151008292e-25, 0.6100566325531255, 0.27139516031925676, 0.007204800838670103, 0.0017296078671290866, 3.516095608561591e-29, 0.01992175288893023, 0.026014160241472187, 7.514772239813612e-07, 0.006167481107732912, 0.037592048033336885])
print('\nTrained pi is:\n',trained_pi)
print("\nVector of differences (ours - trained) is: \n", np.subtract(res, trained_pi))


'''


# print 'Now fitting MMM MODEL \n'
# MMM_LL_VEC = []
# given_e_mat = pd.read_csv("data/given_E12_matrix.csv", index_col=0).values

# for i in range(3):

# print(w[0].e)
# print(w[0].pie)
# print(w[1].e)
# print(w[1].pie)

    # print(MMM_model.e, MMM_model.pie)
    # MMM_LL_VEC.append(MMM_model.log_likelihood(X))
#
#



# df_LL_all_models = pd.DataFrame()
# df_LL_all_models['MMM_model']  = MMM_LL_VEC
# LL_VEC = []
# n = 20
# LDA_12_E = pd.read_csv("data/010519/LDA_12_term_Ematrix.csv", index_col=0).values
# # LDA_12_pi = pd.read_csv("data/010519/LDA_12_topics_matrix.csv")
# print("Now fitting LDA 12 model\n")
# for i in range(n):
#     print(f"Iteration: {i}\n")
#     LDA_12_Model = fit(X, 0.001, 10, LDA_12_E)
#     LL_VEC.append(Theta.log_likelihood(X, LDA_12_Model))
#
# print(LDA_LL_vec)
#
# LL_vec = []
# CTM_12_E = pd.read_csv("data/010519/CTM_15_e.csv", index_col=0).values
# # LDA_12_pi = pd.read_csv("data/010519/LDA_12_topics_matrix.csv")
# print("Now fitting CTM 12 model\n")
# for i in range(n):
#     print(f"Iteration: {i}\n")
#     LDA_12_Model = fit(X, 0.001, 10, LDA_12_E)
#     LL_vec.append(Theta.log_likelihood(X, CTM_12_E))
#
# print(LL_vec)
#
# set_num_of_topics(15)
#
#
#




#
# print("Now fitting MMM 12 model\n")
# MMM_12_E = pd.read_csv("data/010519/LDA_12_term_Ematrix.csv", index_col=0).values


# # print(LDA_Emat)
#
# for i in range(3):
#     LDA_model= fit(X, 0.001, 100, LDA_Emat)[0]
#     print(LDA_model)
#     LDA_LL_vec.append(LDA_model.log_likelihood(X))
#
#
#
# # print "Our CTM pi is:\n ", CTM_res
# print 'LDA ll vector of 20 runs with random pi, and fixed E matrix:\n'
# print LDA_LL_vec
# df_LL_all_models['LDA_model']  = LDA_LL_vec
#
#
# CTM_LL_vec = []
# print 'Now fitting CTM MODEL \n'
# CTM_Emat = pd.read_csv("data/CTM_12_term_matrix.csv", index_col=0).values
# # print(CTM_Emat)
# for i in range(20):
#     CTM_model= fit(X, 0.001, 100, CTM_Emat)[0]
#     CTM_LL_vec.append(CTM_model.log_likelihood(X))
#
#
# # Do exoenonet for representation
# # CTM_res = (np.exp(LDA_model.pie))
#
# # print "Our CTM pi is:\n ", CTM_res
# print 'CTM ll vector of 20 runs with random pi, and fixed E matrix:\n'
# print CTM_LL_vec
#
# df_LL_all_models['CTM_model']  = CTM_LL_vec
# print(df_LL_all_models)
#
# df_LL_all_models.to_csv("data/results_20_runs.csv")

# print(CTM_model.log_likelihood(X))
#
#
# ''' @param sample_data:dict where keys are chromosomes and values are a dict with only 1 key
#         'Sequence' and it's value is list of mutations number
#         @ret : log_probability calculated using current pi, e matrix to see sample data
#     '''
#     def log_probability(self, sample_data):
#         tmp = self.e_mat.T + self.pi
#         joint_prob = logsumexp(tmp, axis=1)
#         mut_counter = mutation_occurences_counter(sample_data)
#         return np.dot(joint_prob, mut_counter)