
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import warnings
import json

num_of_topics = 12
num_of_muts = 96
num_of_samples = 560


warnings.filterwarnings("ignore", message="divide by zero encountered in log")

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
        self.e = np.zeros((num_of_topics, num_of_muts))
        self.pie = np.zeros(num_of_topics)
        self.Emeans = np.zeros((num_of_topics, num_of_muts))



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
        self.e = np.zeros((num_of_topics, num_of_muts))
        for i in range (num_of_topics):
            #generate random vector for each sig
            self.e[i] = np.random.rand(num_of_muts)
            #normalize probabilities
            s = sum(self.e[i])
            self.e[i] /= s

        #get log values of the randomized values
        self.e = np.log(self.e)
        self.pie = np.log(self.pie)
        return self

    def fill_Emeans(self, B) -> int:
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


    def theta_t(self,E_res, fixed_e_flag: bool = False, fixed_pi_flag: bool = False):
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


        # If fixed e flag is False, then e matrix should be updated.
        if not fixed_e_flag:
            # print("fixed e_flag is off, updating e matrix\n")
            for i in range(num_of_muts):
                theta_new.e[:,i] = np.subtract(E_res[:,i], logsumexp(E_res,axis=1))
            theta_new.e[np.isnan(theta_new.e)] = 0

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
        res_array = np.zeros((num_of_samples))
        for j in range(num_of_samples):
            arr = np.zeros(num_of_muts)
            for i in range(array[j].e.shape[1]):
                # print(self.e.T[i])
                arr[i] = logsumexp(array[j].e.T[i] + array[j].pie) + np.log(X[X.columns[j]].loc[i])
            res_array[j] = logsumexp(arr)
        return logsumexp(res_array)

    @staticmethod
    def log_likelihood2(X, array) -> np.array:
        res_array = np.zeros((num_of_samples))
        for j in range(num_of_samples):
            curr_theta = array[j]
            temp = curr_theta.e.T + curr_theta.pie
            joint_prob = logsumexp(temp, axis=1)
            curr_LL = np.dot(joint_prob, X.iloc[:,j])
            # print(curr_LL)

            res_array[j] = curr_LL

        return sum(res_array)


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


def set_num_of_topics(k: int):
    global num_of_topics
    num_of_topics = k
    print(f"Changed num of topics to : {num_of_topics}")
    return

def set_num_of_muts(n: int):
    global num_of_muts
    num_of_muts = n
    print(f"Changed num of muts to : {num_of_muts}")
    return

def set_num_of_samples(m: int):
    global num_of_samples
    num_of_samples = m
    print(f"Changed num of samples to : {num_of_samples}")
    return



def generate_LL_df(model_name: str, k: int, chrom_lst : list, E_mat: pd.DataFrame = None, n: int = 3, fixed_e_flag: bool = False):
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

        print(f"Now training {model_name} model, on full dataset with chrom #{chrom_id} out.\n")
        ll_dict[f'{model_name}_{num_of_topics}_{chrom_id}'] = []

        # For each chromosome generate n results to overcome random data
        for i in range(n):


            if not fixed_e_flag: #Fit de-novo
                print(f"Mode de-novo, Iteration: #{i} : ", end="")
                Model = fit(training_set, 0.001, 50)  # Change input matrix to whichever matrix you want
            else: #Re-fit - meaning muts*topics probabilites matrix (E_mat) remains fixed.
                print(f"Mode refit,  Iteration: #{i} : ", end="")
                Model = fit(training_set, 0.001, 50, E_mat, None, fixed_e_flag, False)

            if (Model == 1):  # if model didn't converge, 1 is appended instead of a negative likelihood score
                print("Did not converrge, appending 1\n")
                ll_dict[f'{model_name}_{num_of_topics}_{chrom_id}'].append(1.0)
            else:
                # Else model converged, check LL and append it.
                print("Model converged, appending LL score: ", end="")
                ll_dict[f'{model_name}_{num_of_topics}_{chrom_id}'].append(Theta.log_likelihood2(test, Model))
                print(ll_dict[f'{model_name}_{num_of_topics}_{chrom_id}'][i])

            print(f"Finished Iteration: {i}\n")

    return pd.DataFrame(ll_dict)


def fix_count(X, len = num_of_muts):

    """
    Takes as input a vector with indices and count how many time each one of them was seen.
    Limited to num_of_muts for now.

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



def fit(X, threshold, max_iterations, init_e : pd.DataFrame = None, init_pie : pd.DataFrame = None,
        fixed_e_flag: bool = False, fixed_pi_flag: bool = False):
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

    ##Needs to be an array - of num_of_samples theta, #Initiate current theta model, randomly
    arr = []
    for i in range(num_of_samples):
        theta_curr = Theta()
        theta_curr.build_random_theta()
        arr.append(theta_curr)

    # If an initial pi matrix wass provided, assign it by order to each theta model (representing a single person)
    if init_pie is not None:
        print("Setting initial pie vecs for each sample\n")
        for i in range(num_of_samples):
            arr[i].pie = np.array(init_pie.iloc[i].values[1:]).astype(np.float64)
            arr[i].pie = np.log(arr[i].pie)

    #  If an initial e matrix wass provided, assign it by order to each theta model (representing a single person)
    if( init_e is not None):
        print("Setting initial E mat for all samples\n")
        for i in range(num_of_samples):
            arr[i].e = np.log(init_e)


    E_res = np.zeros((num_of_topics,num_of_muts))

    for i in range(num_of_samples):
        arr[i].fill_Emeans(X[X.columns[i]])
        np.logaddexp(E_res,arr[i].Emeans,E_res)
    E_res = E_res - np.log(num_of_samples)
    iter = 0
    conv = 0
    E_res1 = np.zeros((num_of_topics,num_of_muts))

    while (conv == 0 and iter < max_iterations):
        arr_old = arr.copy()
        for i in range(num_of_samples):
            theta_prev = arr[i].copy()
        # maximze
            arr[i] = theta_prev.theta_t(E_res, fixed_e_flag, fixed_pi_flag)
        #recalculate E step
            arr[i].fill_Emeans(X[X.columns[i]])
            np.logaddexp(E_res1,arr[i].Emeans,E_res1)
        # print(arr[0].pie)
        E_res1 = E_res1 - np.log(num_of_samples)
        E_res = E_res1.copy()
        # Check
        conv = theta_curr.check_convergence(X, threshold,arr,arr_old)
        iter += 1

    #Check final convergence status
    if(conv == 0):
        print("did not converge!")
        return 1
    else:
        print('Converged after ' + str(iter)+ ' iterations')

    return arr



def upload_model(terms_path,topics_path,num_of_topics, model_name = None):


    #enter line that says how many topics are used
    set_num_of_topics((num_of_topics))
    terms = pd.read_csv(terms_path,index_col=0)
    topics = pd.read_csv(topics_path,index_col=0)

    if(model_name is not None and str.lower(model_name) == 'ctm'):
        terms = pd.read_csv(terms_path, index_col=0).reset_index()
        topics = pd.read_csv(topics_path, index_col=0).reset_index()
        print(terms)
        terms = np.exp(terms)
    arr = []
    for i in range(num_of_samples):
        theta = Theta()
        theta.pie = np.log(topics.loc[topics.index[i]].values)
        theta.e = np.log(terms.values)
        arr.append(theta)
    return arr

# model = upload_model("mmm_outputs/LDA_12_term_matrix.csv","mmm_outputs/LDA_12_topics_matrix.csv",12)


def transform_chr_to_np(chr_dict):
    lst = []
    # print list(chr_dict.keys())
    for chr in chr_dict:
        # print "Transforming\n"
        # print(chr_dict[chr]['Sequence'])
        lst.extend(chr_dict[chr]['Sequence'])

    return np.array(lst)

