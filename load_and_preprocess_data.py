import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.model_selection import KFold

class TrainDataInfo:
    def __init__(self, file_map, country_names, country_counts, sampling_probs, weight_scalars):
        '''
        Make copies of all the inputs, then we'll define a few accessor methods
        '''
        self.filename_map = file_map.copy()     # dict(country_name -> filename_list)
        self.country_names = country_names[:]   # sorted country names
        self.sampling_probs = sampling_probs[:] # sampling probabilities (sorted by name)
        self.weight_scalars = weight_scalars[:] # loss fn weight scalars (sorted by name)
        
        self.country_counts = [len(self.filename_map[c]) for c in self.country_names]
        
    # The vectors above are stored in country-name sorted order, even though filename_map
    # doesn't have any ordering.  But it is still nice to be able to query values by name.
    def get_country_count(self, country):
        return self.country_counts[self.country_names.index(country)]
    
    def get_sampling_prob(self, country):
        return self.sampling_probs[self.country_names.index(country)]
    
    def get_weight_scalars(self, country):
        return self.weight_scalars[self.country_names.index(country)]
   
    
def collect_filename_mapping(train_folder):
    file_map = defaultdict(lambda: list())
    country_counts = []

    for root,subs,files in os.walk(train_folder):
        country = os.path.split(root.lstrip(train_folder))[1]
        if len(country.strip()) == 0:
            continue

        for f in files:
            if not os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            file_map[country].append(os.path.join(root, f))

    # Sorted names, which will match the sampling probs and weight vectors
    country_names = sorted(file_map.keys())
    country_counts = [len(file_map[c]) for c in country_names]
    for country in country_names:
        file_map[country] = sorted(file_map[country])
        
    # We're going to sample from the classes proportional to sqrt(N)
    sampling_probs = np.sqrt(np.array(country_counts)) 
    sampling_probs = sampling_probs / np.sum(sampling_probs)

    # When we actually do training, we want to scale gradient updates by 1/sqrt(N)
    weight_scalars = 1.0 / sampling_probs
    weight_scalars = weight_scalars / np.mean(weight_scalars)
    #weight_scalars = np.clip(weight_scalars / np.mean(weight_scalars), 0.7, 2.0)

    return TrainDataInfo(file_map, country_names, country_counts, sampling_probs, weight_scalars)


def get_kfold_train_test(ds_size, shards=5, shard_test_index=None, shuffle=True):
    kf = KFold(n_splits=shards, shuffle=shuffle, random_state=31415926)
    if shard_test_index is None:
        return [(train,test) for train,test in kf.split(range(ds_size))]
    else:
        # We're looking for a specific split.  But kf.split() produces 
        # a generator, so you have to iterate if you want a specific one
        for i,traintest in enumerate(kf.split(range(ds_size))):
            if i == shard_test_index:
                return traintest
        else:
            raise Exception('Something went terribly wrong!')
    
    
def distributed_train_test_split(all_fn_map, shards=5, shard_test_index=0):
    """
    We are going to do a test-train split but within each class/country, so that we can
    manage sampling rates (oversampling Armenia, Australia, undersampling Russia, etc)
    """
    train_fn_map = defaultdict(lambda: [])
    val_fn_map = defaultdict(lambda: [])
    
    for country,fn_list in all_fn_map.items():
        n_files = len(fn_list)
        train_indices, test_indices = get_kfold_train_test(n_files, shards, shard_test_index)
        train_fn_map[country] = [all_fn_map[country][i] for i in train_indices]
        val_fn_map[country] = [all_fn_map[country][i] for i in test_indices]
    
    return train_fn_map, val_fn_map


def get_country_sample(n_country, names, sampling_probs, *, with_replace):
    """
    For triplet loss, need 2 w/o replacement; general training needs batch size w/ replacement
    """
    if not with_replace and n_country > len(names):
        raise Exception(f'Cannot take {n_country} samples w/o replacement, list size is {len(names)}')
        
    idx_choice = np.random.choice(len(names), size=n_country, p=sampling_probs, replace=with_replace)
    
    # We actually need to return the number of images to pull from each country
    counts = Counter([names[i] for i in idx_choice])
    print(counts)
    return [(country, ct) for country,ct in counts.items()]


def get_image_fn_sample(fn_map, n_img, country, *, with_replace):
    """
    Given the filename mapping, select {n_img} images for the specified country
    """
    if not with_replace and n_img > len(fn_map[country]):
        raise Exception(f'Cannot take {count} samples w/o replacement, list size is {len(names)}')
        
    return np.random.choice(fn_map[country], size=n_img, replace=with_replace)
    
    

def display_rel_counts(train_info):
        
    trunc_names = [n[:12] for n in train_info.country_names]
    counts = train_info.country_counts 
    sample_rates = train_info.sampling_probs
    weights = train_info.weight_scalars
    
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
    
def collect_filename_mapping(train_folder):
    file_map = defaultdict(lambda: list())
    country_counts = []

    for root,subs,files in os.walk(train_folder):
        country = os.path.split(root.lstrip(train_folder))[1]
        if len(country.strip()) == 0:
            continue

        for f in files:
            if not os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            file_map[country].append(os.path.join(root, f))

    # Sorted names, which will match the sampling probs and weight vectors
    country_names = sorted(file_map.keys())
    country_counts = [len(file_map[c]) for c in country_names]
    for country in country_names:
        file_map[country] = sorted(file_map[country])
        
    # We're going to sample from the classes proportional to sqrt(N)
    sampling_probs = np.sqrt(np.array(country_counts)) 
    sampling_probs = sampling_probs / np.sum(sampling_probs)

    # When we actually do training, we want to scale gradient updates by 1/sqrt(N)
    weight_scalars = 1.0 / sampling_probs
    weight_scalars = weight_scalars / np.mean(weight_scalars)
    #weight_scalars = np.clip(weight_scalars / np.mean(weight_scalars), 0.7, 2.0)

    return TrainDataInfo(file_map, country_names, country_counts, sampling_probs, weight_scalars)


def get_kfold_train_test(ds_size, shards=5, shard_test_index=None, shuffle=True):
    kf = KFold(n_splits=shards, shuffle=shuffle, random_state=31415926)
    if shard_test_index is None:
        return [(train,test) for train,test in kf.split(range(ds_size))]
    else:
        # We're looking for a specific split.  But kf.split() produces 
        # a generator, so you have to iterate if you want a specific one
        for i,traintest in enumerate(kf.split(range(ds_size))):
            if i == shard_test_index:
                return traintest
        else:
            raise Exception('Something went terribly wrong!')
    
    
def distributed_train_test_split(all_fn_map, shards=5, shard_test_index=0):
    """
    We are going to do a test-train split but within each class/country, so that we can
    manage sampling rates (oversampling Armenia, Australia, undersampling Russia, etc)
    """
    train_fn_map = defaultdict(lambda: [])
    val_fn_map = defaultdict(lambda: [])
    
    for country,fn_list in all_fn_map.items():
        n_files = len(fn_list)
        train_indices, test_indices = get_kfold_train_test(n_files, shards, shard_test_index)
        train_fn_map[country] = [all_fn_map[country][i] for i in train_indices]
        val_fn_map[country] = [all_fn_map[country][i] for i in test_indices]
    
    return train_fn_map, val_fn_map


def get_country_sample(n_country, names, sampling_probs, *, with_replace):
    """
    For triplet loss, need 2 w/o replacement; general training needs batch size w/ replacement
    """
    if not with_replace and n_country > len(names):
        raise Exception(f'Cannot take {n_country} samples w/o replacement, list size is {len(names)}')
        
    idx_choice = np.random.choice(len(names), size=n_country, p=sampling_probs, replace=with_replace)
    
    # We actually need to return the number of images to pull from each country
    counts = Counter([names[i] for i in idx_choice])
    print(counts)
    return counts


def get_image_fn_sample(fn_map, n_img, country, *, with_replace):
    """
    Given the filename mapping, select {n_img} images for the specified country
    """
    if not with_replace and n_img > len(fn_map[country]):
        raise Exception(f'Cannot take {count} samples w/o replacement, list size is {len(names)}')
        
    return np.random.choice(fn_map[country], size=n_img, replace=with_replace)
    
    

def display_rel_counts(train_info):
        
    trunc_names = [n[:12] for n in train_info.country_names]
    counts = train_info.country_counts 
    sample_rates = train_info.sampling_probs
    weights = train_info.weight_scalars
    
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
    
def collect_filename_mapping(train_folder):
    file_map = defaultdict(lambda: list())
    country_counts = []

    for root,subs,files in os.walk(train_folder):
        country = os.path.split(root.lstrip(train_folder))[1]
        if len(country.strip()) == 0:
            continue

        for f in files:
            if not os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            file_map[country].append(os.path.join(root, f))

    # Sorted names, which will match the sampling probs and weight vectors
    country_names = sorted(file_map.keys())
    country_counts = [len(file_map[c]) for c in country_names]
    for country in country_names:
        file_map[country] = sorted(file_map[country])
        
    # We're going to sample from the classes proportional to sqrt(N)
    sampling_probs = np.sqrt(np.array(country_counts)) 
    sampling_probs = sampling_probs / np.sum(sampling_probs)

    # When we actually do training, we want to scale gradient updates by 1/sqrt(N)
    weight_scalars = 1.0 / sampling_probs
    weight_scalars = weight_scalars / np.mean(weight_scalars)
    #weight_scalars = np.clip(weight_scalars / np.mean(weight_scalars), 0.7, 2.0)

    return TrainDataInfo(file_map, country_names, country_counts, sampling_probs, weight_scalars)


def get_kfold_train_test(ds_size, shards=5, shard_test_index=None, shuffle=True):
    kf = KFold(n_splits=shards, shuffle=shuffle, random_state=31415926)
    if shard_test_index is None:
        return [(train,test) for train,test in kf.split(range(ds_size))]
    else:
        # We're looking for a specific split.  But kf.split() produces 
        # a generator, so you have to iterate if you want a specific one
        for i,traintest in enumerate(kf.split(range(ds_size))):
            if i == shard_test_index:
                return traintest
        else:
            raise Exception('Something went terribly wrong!')
    
    
def distributed_train_test_split(all_fn_map, shards=5, shard_test_index=0):
    """
    We are going to do a test-train split but within each class/country, so that we can
    manage sampling rates (oversampling Armenia, Australia, undersampling Russia, etc)
    """
    train_fn_map = defaultdict(lambda: [])
    val_fn_map = defaultdict(lambda: [])
    
    for country,fn_list in all_fn_map.items():
        n_files = len(fn_list)
        train_indices, test_indices = get_kfold_train_test(n_files, shards, shard_test_index)
        train_fn_map[country] = [all_fn_map[country][i] for i in train_indices]
        val_fn_map[country] = [all_fn_map[country][i] for i in test_indices]
    
    return train_fn_map, val_fn_map


def get_country_sample(n_country, names, sampling_probs, *, with_replace):
    """
    For triplet loss, need 2 w/o replacement; general training needs batch size w/ replacement
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
    """
    if not with_replace and n_img > len(fn_map[country]):
        raise Exception(f'Cannot take {count} samples w/o replacement, list size is {len(names)}')
        
    return np.random.choice(fn_map[country], size=n_img, replace=with_replace)
    
    

def display_rel_counts(train_info):
        
    trunc_names = [n[:12] for n in train_info.country_names]
    counts = train_info.country_counts 
    sample_rates = train_info.sampling_probs
    weights = train_info.weight_scalars
    
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
    
def collect_filename_mapping(train_folder):
    file_map = defaultdict(lambda: list())
    country_counts = []

    for root,subs,files in os.walk(train_folder):
        country = os.path.split(root.lstrip(train_folder))[1]
        if len(country.strip()) == 0:
            continue

        for f in files:
            if not os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            file_map[country].append(os.path.join(root, f))

    # Sorted names, which will match the sampling probs and weight vectors
    country_names = sorted(file_map.keys())
    country_counts = [len(file_map[c]) for c in country_names]
    for country in country_names:
        file_map[country] = sorted(file_map[country])
        
    # We're going to sample from the classes proportional to sqrt(N)
    sampling_probs = np.sqrt(np.array(country_counts)) 
    sampling_probs = sampling_probs / np.sum(sampling_probs)

    # When we actually do training, we want to scale gradient updates by 1/sqrt(N)
    weight_scalars = 1.0 / sampling_probs
    #weight_scalars = weight_scalars / min(weight_scalars)
    weight_scalars = np.clip(weight_scalars / min(weight_scalars), 1.0, 2.0)

    return TrainDataInfo(file_map, country_names, country_counts, sampling_probs, weight_scalars)


def get_kfold_train_test(ds_size, shards=5, shard_test_index=None, shuffle=True):
    kf = KFold(n_splits=shards, shuffle=shuffle, random_state=31415926)
    if shard_test_index is None:
        return [(train,test) for train,test in kf.split(range(ds_size))]
    else:
        # We're looking for a specific split.  But kf.split() produces 
        # a generator, so you have to iterate if you want a specific one
        for i,traintest in enumerate(kf.split(range(ds_size))):
            if i == shard_test_index:
                return traintest
        else:
            raise Exception('Something went terribly wrong!')
    
    
def distributed_train_test_split(all_fn_map, shards=5, shard_test_index=0):
    """
    We are going to do a test-train split but within each class/country, so that we can
    manage sampling rates (oversampling Armenia, Australia, undersampling Russia, etc)
    """
    train_fn_map = defaultdict(lambda: [])
    val_fn_map = defaultdict(lambda: [])
    
    for country,fn_list in all_fn_map.items():
        n_files = len(fn_list)
        train_indices, test_indices = get_kfold_train_test(n_files, shards, shard_test_index)
        train_fn_map[country] = [all_fn_map[country][i] for i in train_indices]
        val_fn_map[country] = [all_fn_map[country][i] for i in test_indices]
    
    return train_fn_map, val_fn_map


def get_country_sample(n_country, names, sampling_probs, *, with_replace):
    """
    For triplet loss, need 2 w/o replacement; general training needs batch size w/ replacement
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
    """
    if not with_replace and n_img > len(fn_map[country]):
        raise Exception(f'Cannot take {count} samples w/o replacement, list size is {len(names)}')
        
    return np.random.choice(fn_map[country], size=n_img, replace=with_replace)
    
    

def display_rel_counts(train_info):
        
    trunc_names = [n[:12] for n in train_info.country_names]
    counts = train_info.country_counts 
    sample_rates = train_info.sampling_probs
    weights = train_info.weight_scalars
    
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
    
def collect_filename_mapping(train_folder):
    file_map = defaultdict(lambda: list())
    country_counts = []

    for root,subs,files in os.walk(train_folder):
        country = os.path.split(root.lstrip(train_folder))[1]
        if len(country.strip()) == 0:
            continue

        for f in files:
            if not os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            file_map[country].append(os.path.join(root, f))

    # Sorted names, which will match the sampling probs and weight vectors
    country_names = sorted(file_map.keys())
    country_counts = [len(file_map[c]) for c in country_names]
    for country in country_names:
        file_map[country] = sorted(file_map[country])
        
    # We're going to sample from the classes proportional to sqrt(N)
    sampling_probs = np.sqrt(np.array(country_counts)) 
    sampling_probs = sampling_probs / np.sum(sampling_probs)

    # When we actually do training, we want to scale gradient updates by 1/sqrt(N)
    weight_scalars = 1.0 / sampling_probs
    #weight_scalars = weight_scalars / min(weight_scalars)
    weight_scalars = np.clip(weight_scalars / np.mean(weight_scalars), 0.8, 2.0)

    return TrainDataInfo(file_map, country_names, country_counts, sampling_probs, weight_scalars)


def get_kfold_train_test(ds_size, shards=5, shard_test_index=None, shuffle=True):
    kf = KFold(n_splits=shards, shuffle=shuffle, random_state=31415926)
    if shard_test_index is None:
        return [(train,test) for train,test in kf.split(range(ds_size))]
    else:
        # We're looking for a specific split.  But kf.split() produces 
        # a generator, so you have to iterate if you want a specific one
        for i,traintest in enumerate(kf.split(range(ds_size))):
            if i == shard_test_index:
                return traintest
        else:
            raise Exception('Something went terribly wrong!')
    
    
def distributed_train_test_split(all_fn_map, shards=5, shard_test_index=0):
    """
    We are going to do a test-train split but within each class/country, so that we can
    manage sampling rates (oversampling Armenia, Australia, undersampling Russia, etc)
    """
    train_fn_map = defaultdict(lambda: [])
    val_fn_map = defaultdict(lambda: [])
    
    for country,fn_list in all_fn_map.items():
        n_files = len(fn_list)
        train_indices, test_indices = get_kfold_train_test(n_files, shards, shard_test_index)
        train_fn_map[country] = [all_fn_map[country][i] for i in train_indices]
        val_fn_map[country] = [all_fn_map[country][i] for i in test_indices]
    
    return train_fn_map, val_fn_map


def get_country_sample(n_country, names, sampling_probs, *, with_replace):
    """
    For triplet loss, need 2 w/o replacement; general training needs batch size w/ replacement
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
    """
    if not with_replace and n_img > len(fn_map[country]):
        raise Exception(f'Cannot take {count} samples w/o replacement, list size is {len(names)}')
        
    return np.random.choice(fn_map[country], size=n_img, replace=with_replace)
    
    

def display_rel_counts(train_info):
        
    trunc_names = [n[:12] for n in train_info.country_names]
    counts = train_info.country_counts 
    sample_rates = train_info.sampling_probs
    weights = train_info.weight_scalars
    
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
