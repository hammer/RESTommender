import pandas
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy
import pickle
import random

class ProbabilisticMatrixFactorization():
    """
    Attributes
    * latent_d
    * learning_rate, regularization_strength
    * ratings
    * num_users, num_items
    * users, items
    * iterations, converged

    Methods
    * cost
    * update_batch
    * update_online
    """
    def __init__(self, rating_df, latent_d=1):
        self.latent_d = latent_d
        self.learning_rate = .1 # alpha
        self.regularization_strength = 1.0 # lambda

        self.ratings = numpy.array(rating_df).astype(float)

        self.num_users = int(numpy.max(self.ratings[:, 0]) + 1)
        self.num_items = int(numpy.max(self.ratings[:, 1]) + 1)

        # initialize latent features for users and items
        self.users = numpy.random.normal(size=(self.num_users, self.latent_d))
        self.items = numpy.random.normal(size=(self.num_items, self.latent_d))

        self.iterations = 0
        self.converged = False
        self.shuffle = None


    def cost(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        # empirical risk term
        squared_error = 0
        for i, j, rating in self.ratings:
            r_hat = numpy.sum(users[i] * items[j])
            squared_error += (rating - r_hat)**2
        rmse = numpy.sqrt(squared_error / self.ratings.shape[0])

        # regularization term
        L2_norm = 0
        L2_norm += numpy.sum(numpy.sqrt(numpy.sum(numpy.square(users), axis=1)))
        L2_norm += numpy.sum(numpy.sqrt(numpy.sum(numpy.square(items), axis=1)))

        return rmse + self.regularization_strength * L2_norm
        
        
    def update_batch(self):
        self.iterations += 1

        # updates_o holds updates to the latent features of users
        # updates_d holds updates to the latent features of items
        updates_o = numpy.zeros((self.num_users, self.latent_d))
        updates_d = numpy.zeros((self.num_items, self.latent_d))        

        # batch update: run through all ratings for each iteration
        for i, j, rating in self.ratings:
            # r_hat is the predicted rating for user i on item j
            r_hat = numpy.sum(self.users[i] * self.items[j])

            # update each feature according to weight accurracy            
            updates_o[i, :] += self.items[j, :] * (rating - r_hat)
            updates_d[j, :] += self.users[i, :] * (rating - r_hat)

        # calculate new parameters
        user_updates = self.learning_rate * (updates_o + self.regularization_strength * self.users)
        item_updates = self.learning_rate * (updates_d + self.regularization_strength * self.items)

        # converge if cost changes by less than .1 or if learning rate goes below 1e-10
        # speed up by 1.25x if improving, slow down by 0.5x if not improving
        old_cost = self.cost()
        new_cost = self.cost(self.users - user_updates, self.items - item_updates)

        # if the new latent feature vectors are better, keep the updates, and increase the learning rate (i.e. momentum)
        if new_cost < old_cost:
            # make weight changes permanent
            self.users -= user_updates
            self.items -= item_updates

            self.learning_rate *= 1.25

            if (old_cost - new_cost) / old_cost < .001:
                self.converged = True
        else:
            self.learning_rate *= .5

        if self.learning_rate < 1e-8:
            self.converged = True

    def update_online(self):
        # shuffle up the ratings datan
        if not self.shuffle:
            self.shuffle = range(self.ratings.shape[0])
            random.shuffle(self.shuffle)

        self.iterations += 1
        eta = self.learning_rate / (1 + self.iterations / self.ratings.shape[0])

        # updates_o holds updates to the latent features of users
        # updates_d holds updates to the latent features of items
        updates_o = numpy.zeros((self.num_users, self.latent_d))
        updates_d = numpy.zeros((self.num_items, self.latent_d))        

        # get a random example
        i, j, rating = self.ratings[self.shuffle.pop()]

        # r_hat is the predicted rating for user i on item j
        r_hat = numpy.sum(self.users[i] * self.items[j])

        # update each feature according to weight accurracy            
        updates_o[i, :] += self.items[j, :] * (rating - r_hat)
        updates_d[j, :] += self.users[i, :] * (rating - r_hat)

        # calculate new parameters
        user_updates = eta * (updates_o + self.regularization_strength * self.users)
        item_updates = eta * (updates_d + self.regularization_strength * self.items)

        # apply updates
        self.users -= user_updates
        self.items -= item_updates

        # stop after a single pass
        if self.iterations >= self.ratings.shape[0]:
            self.converged = True


if __name__ == "__main__":
    # read in ratings data
    raw_data = pandas.read_csv('/Users/hammer/codebox/RESTommender/notebook/ml-100k/u.data', header=None, sep='\t')
    raw_data.columns = ["user_id", "item_id", "rating", "timestamp"]
    raw_data[["user_id", "item_id"]] -= 1 # make ids 0-indexed
    ratings_mean = numpy.mean(raw_data["rating"])
    raw_data["rating"] -= ratings_mean
    ratings = raw_data[["user_id", "item_id", "rating"]]

    # construct training, cv, and test sets
    nrows = ratings.shape[0]
    held_out_size = int(nrows * 0.20)
    held_out = ratings.ix[random.sample(ratings.index, held_out_size)]
    training_set = ratings.ix[[i for i in ratings.index if i not in held_out.index]]
    cv_set = held_out[:int(held_out_size/2)]
    test_set = held_out[int(held_out_size/2):]

    # fit model
    pmf = ProbabilisticMatrixFactorization(training_set, latent_d=10)
    while not pmf.converged:
        if pmf.iterations % 10000 == 0:
            print "Iteration: %s" % pmf.iterations
            print "Cost: %s" % pmf.cost()
            #print "Learning rate: %s" % pmf.learning_rate
        pmf.update_online()
    
    # save learned parameters
    run = random.choice(range(1000))
    pickle.dump(pmf.users, open('users%s.pkl' % run, 'w'))
    pickle.dump(pmf.items, open('items%s.pkl' % run, 'w'))

    # evaluate performance on cv set for hyperparameter tuning
    squared_error = 0    
    for i, j, rating in cv_set.as_matrix():
      r_hat = numpy.sum(pmf.users[i, :] * pmf.items[j, :])
      squared_error += (rating - r_hat)**2
    rmse = numpy.sqrt(squared_error / cv_set.shape[0])
    print "RMSE: %s" % rmse
