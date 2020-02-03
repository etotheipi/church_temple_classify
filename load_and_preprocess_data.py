import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.model_selection import KFold


class TrainDataInfo:
    def __init__(self, train_folder, kfold_splits=5):
        '''
        Given a folder, index all images there and various statistics about them
        '''
        if not os.path.isdir(train_folder):
            raise NotADirectoryError(f'Folder "{train_folder}" does not exist')
        
        # This is a list of all the members we're going to have at the end of __init__
        self.filename_map = None
        self.country_names = None
        self.country_counts = None
        self.sampling_probs = None
        self.weight_scalars = None
        self.traintest_splits = []
        
        self.filename_map = defaultdict(lambda: list())
        for root,subs,files in os.walk(train_folder):
            country = os.path.split(root.lstrip('./').rstrip('/'))[-1]
            if len(country.strip()) == 0:
                continue

            for f in files:
                if not os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue

                self.filename_map[country].append(os.path.join(root, f))

        # Sort all the names and counts, will do the same later sampling probs, wgts
        self.country_names = sorted(self.filename_map.keys())
        self.country_counts = [len(self.filename_map[c]) for c in self.country_names]
        for country in self.country_names:
            self.filename_map[country] = sorted(self.filename_map[country])
            
        # We're going to sample from the classes proportional to sqrt(N)
        self.sampling_probs = np.sqrt(np.array(self.country_counts)) 
        self.sampling_probs = self.sampling_probs / np.sum(self.sampling_probs)

        # When we actually do training, we want to scale gradient updates by 1/sqrt(N)
        self.weight_scalars = 1.0 / self.sampling_probs
        self.weight_scalars = self.weight_scalars / np.mean(self.weight_scalars)
        self.weight_scalars = np.clip(self.weight_scalars, 0.8, 2.0)
        
        # Precalculate the train-test splits, for k-fold cross val
        for i in range(kfold_splits):
            train,test = self._distributed_train_test_split(shards=kfold_splits, split_index=i)
            self.traintest_splits.append((train, test))

        
    # The vectors above are stored in country-name sorted order, even though filename_map
    # doesn't have any ordering.  But it is still nice to be able to query values by country
    def get_country_count(self, country):
        return self.country_counts[self.country_names.index(country)]
    
    def get_sampling_prob(self, country):
        return self.sampling_probs[self.country_names.index(country)]
    
    def get_weight_scalars(self, country):
        return self.weight_scalars[self.country_names.index(country)]
   

    def sample_country(self, uniform=False):
        if uniform:
            return np.random.choice(self.country_names)
        else:
            return np.random.choice(self.country_names, p=self.sampling_probs)
        
    def sample_filename(self, file_map=None):
        """
        Use self.sampling_probs to get the correct country sampling rates
        File_map is the specific train or test split set, so use that to get the sample
        """
        if file_map is None:
            file_map = self.filename_map
        
        country = self.sample_country()
        path_idx = np.random.choice(len(file_map[country]))
        return file_map[country][path_idx]
        
    def triplet_sample(self, file_map=None, uniform=False):
        """
        Use self.sampling_probs to get the correct country sampling rates
        File_map is the specific train or test split set, so use that to get the sample
        """
        if file_map is None:
            file_map = self.filename_map
        
        sprobs = self.sampling_probs
        if uniform:
            sprobs = np.ones(shape=(len(sprobs)), dtype='float64') / len(sprobs)
            
        c1, c2 = np.random.choice(self.country_names, size=2, p=sprobs, replace=False).tolist()
        f_anchor, f_pos = np.random.choice(file_map[c1], size=2, replace=False).tolist()
        f_neg = np.random.choice(file_map[c2])
        
        return (f_anchor, f_pos, f_neg)
        
    
    def _get_kfold_train_test(self, ds_size, shards=5, split_index=0, shuffle=True):
        """
        We're want a specific split.  But kf.split() produces  a generator, so iterate if
        you want a specific one (could use itertools but seems overkill here)
        """
        kf = KFold(n_splits=shards, shuffle=shuffle, random_state=31415926)
        for i,traintest in enumerate(kf.split(range(ds_size))):
            if i == split_index:
                return traintest
        else:
            raise Exception('Something went terribly wrong!')


    def _distributed_train_test_split(self, shards=5, split_index=0):
        """
        We are going to do a test-train split but within each class/country, so that we can
        manage sampling rates (oversampling Armenia, Australia, undersampling Russia, etc)
        """
        train_fn_map = defaultdict(lambda: [])
        val_fn_map = defaultdict(lambda: [])

        for country,fn_list in self.filename_map.items():
            n_files = len(fn_list)
            train_indices, test_indices = self._get_kfold_train_test(n_files, shards, split_index)
            train_fn_map[country] = [self.filename_map[country][i] for i in train_indices]
            val_fn_map[country] = [self.filename_map[country][i] for i in test_indices]

        return train_fn_map, val_fn_map
    
    
    def display_rel_counts(self):
        trunc_names = [n[:12] for n in self.country_names]
        counts = self.country_counts 
        sample_rates = self.sampling_probs
        weights = self.weight_scalars

        fig,axs = plt.subplots(1, 3, figsize=(15,5))

        sns.barplot(trunc_names, counts, ax=axs[0])
        axs[0].plot([-1, len(counts)], [min(counts)]*2, 'r-.')
        axs[0].plot([-1, len(counts)], [max(counts)]*2, 'r-.')
        axs[0].set_title('Number of Files per Class')
        axs[0].set_ylabel('Raw Image/File Count')
        for tick in axs[0].get_xticklabels():
            tick.set_rotation(90)

        sns.barplot(trunc_names, sample_rates, ax=axs[1])
        axs[1].plot([-1, len(counts)], [min(sample_rates)]*2, 'r-.')
        axs[1].plot([-1, len(counts)], [max(sample_rates)]*2, 'r-.')
        axs[1].set_title('Sampling Probabilities for Training')
        axs[1].set_ylabel('Sampling Probability')
        for tick in axs[1].get_xticklabels():
            tick.set_rotation(90)

        sns.barplot(trunc_names, weights, ax=axs[2])
        axs[2].set_title('Loss Function Weighting')
        axs[2].set_ylabel('Relative weight')
        axs[2].plot([-1, len(counts)], [1.0, 1.0], 'r-.')
        for tick in axs[2].get_xticklabels():
            tick.set_rotation(90)


            
'''
def get_country_sample(n_country, names, sampling_probs, *, with_replace):
    """
    For triplet loss, need 2 no replacement; general training needs batch size w/ replacement
    This should be called with names and probs from the master train_info object
    """
    if not with_replace and n_country > len(names):
        raise Exception(f'Cannot take {n_country} samples w/o replacement, list size is {len(names)}')

    idx_choice = np.random.choice(len(names), size=n_country, p=sampling_probs, replace=with_replace)

    # We actually need to return the number of images to pull from each country
    counts = Counter([names[i] for i in idx_choice])
    return counts


def get_image_fn_sample(fn_map, n_img, country, *, with_replace):
    """
    Given the filename mapping, select {n_img} images for the specified country
    This should be called with training set of the split dataset
    """
    if not with_replace and n_img > len(fn_map[country]):
        raise Exception(f'Cannot take {count} samples w/o replacement, list size is {len(names)}')

    return np.random.choice(fn_map[country], size=n_img, replace=with_replace)
'''
