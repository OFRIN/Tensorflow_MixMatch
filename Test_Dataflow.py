
from utils.Timer import *
from utils.CIFAR10 import *
from utils.Dataflow import *

if __name__ == '__main__':
    timer = Timer()

    timer.tik()
    train_labeled_dataset, train_unlabeled_dataset, _, _ = get_dataset_cifar10(250)

    print(len(train_labeled_dataset))
    print(len(train_unlabeled_dataset))
    print('dataset {}ms'.format(timer.tok()))
    
    train_labeled_ds = generate_labeled_dataflow(train_labeled_dataset, {
        'augmentors' : [
            Weakly_Augment(),
        ],

        'shuffle' : True,
        'remainder' : False,
    
        'batch_size' : 64,

        'num_prefetch_for_dataset' : 2,
        'num_prefetch_for_batch' : 2,

        'number_of_cores' : 1,
    })
    train_labeled_ds.reset_state()
    train_labeled_iterator = train_labeled_ds.get_data()
    
    train_unlabeled_ds = generate_unlabeled_dataflow(train_unlabeled_dataset, {
        'shuffle' : True,
        'remainder' : False,

        'K' : 3,
        'batch_size' : 64,
        
        'num_prefetch_for_dataset' : 2,
        'num_prefetch_for_batch' : 2,

        'number_of_cores' : 1,
        'augment_func' : Weakly_Augment_func,
    })
    train_unlabeled_ds.reset_state()
    train_unlabeled_iterator = train_unlabeled_ds.get_data()
    
    while True:
        try:
            labeled_dataset = next(train_labeled_iterator)
        except StopIteration:
            print('labeled dataset - stop iteration')
            train_labeled_iterator = train_labeled_ds.get_data()

        try:
            unlabeled_dataset = next(train_unlabeled_iterator)
        except StopIteration:
            print('unlabeled dataset - stop iteration')
            train_unlabeled_iterator = train_unlabeled_ds.get_data()

        print(len(labeled_dataset), len(unlabeled_dataset), unlabeled_dataset[0].shape)
    
