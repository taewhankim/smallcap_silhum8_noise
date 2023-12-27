import torch
from torch.utils.data import Dataset, DataLoader

class datasets(Dataset):
    def __init__(self, captions):
        self.caption = captions
        # detail_prompt = "8k dlsr, highres, masterpiece, photoism, perfect ligthing, "

        prompt_cap = ["8k dlsr, highres, masterpiece, photoism, perfect ligthing, "+i for i in captions]
        # last_config = [i+" hard rim lighting photography--beta --beta --upbeta" for i in prompt_cap]
        self.re_cap = prompt_cap
    def __len__(self):
        return len(self.re_cap)

    def __getitem__(self, idx):
        re_cap = self.re_cap[idx]

        return re_cap


def create_loader(dataset, batch_size, num_workers):
    # train,val, test = dataset
    # train_dataset = datasets(train)
    # val_dataset = datasets(val)
    # test_dataset = datasets(test)
    train_dataset = datasets(dataset[0])

    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    except:
        train_sampler = None
        test_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,
        drop_last=False,
    )

    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     shuffle=False,
    #     drop_last=False,
    # )
    #
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     sampler=test_sampler,
    #     shuffle=False,
    #     drop_last=False,
    # )
    #
    # return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader