import torch


class Stage1_RN50(torch.nn.Module):
    def __init__(self, emb_dim: int, one_proj: bool, adapter: bool, data_dir: str):
        super().__init__()
        self.emb_dim = emb_dim
        self.one_proj = one_proj
        self.adapter = adapter

        attr_embs = torch.load(f"{data_dir}/attr_embs.pth")

        if self.one_proj:
            self.num_proj = 1
        else:
            self.num_proj = len(attr_embs)

        self.pool = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.emb_dim, self.emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.emb_dim, self.emb_dim),
                )
                for i in range(self.num_proj)
            ]
        )

    def forward(self, sample):
        out_dict = {}
        img_vector = sample["img"].to(torch.float32)
        out_img_a = []
        for i in range(self.num_proj):
            pool = self.pool[i](img_vector)
            if self.adapter:
                pool = 0.2 * pool + 0.8 * img_vector
            out_img_a.append(pool)
        region_proj_embs = torch.cat(out_img_a, dim=1).view(
            -1, self.num_proj, self.emb_dim
        )
        out_dict["region_proj_embs"] = region_proj_embs

        return out_dict
