# This file support run_train_model, run_IF_exp, and run_MIS_exp
import transformers
import torch
import torch.nn as nn
import re
import logging
from src.loss import multi_loss_fn


# --------------------Model and Tokenizer-------------------------------------------------------
# Model and tokenizer downloading functions are altered from Mitchel et. al. (2022), MEND (https://github.com/eric-mitchell/mend;https://openreview.net/pdf?id=0DcZxeWfOPt)
class CastModule(nn.Module):
    def __init__(self, module: nn.Module, in_cast: torch.dtype = torch.float32, out_cast: torch.dtype = None):
        super().__init__()

        self.underlying = module
        self.in_cast = in_cast
        self.out_cast = out_cast

    def cast(self, obj, dtype):
        if dtype is None:
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.to(dtype)
        else:
            return obj

    def forward(self, *args, **kwargs):
        args = tuple(self.cast(a, self.in_cast) for a in args)
        kwargs = {k: self.cast(v, self.in_cast) for k, v in kwargs.items()}
        outputs = self.underlying(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = self.cast(outputs, self.out_cast)
        elif isinstance(outputs, tuple):
            outputs = tuple(self.cast(o, self.out_cast) for o in outputs)
        else:
            raise RuntimeError(f"Not sure how to cast type {type(outputs)}")
        return outputs

    def extra_repr(self):
        return f"in_cast: {self.in_cast}\nout_cast: {self.out_cast}"


def scr():
    scr_dir = "../src"


def get_model(tokenizer, model_name, transformers, model_class_name, model_pt, dropout, model_inner_params, no_grad_layers, base_dir="", half=None):
    LOG = logging.getLogger(__name__)
    ModelClass = getattr(transformers, model_class_name)
    LOG.info(
        f"Loading model class {ModelClass} with name {model_name} from cache dir {scr()}")
    model = ModelClass.from_pretrained(model_name, cache_dir=scr())

    if model_pt is not None:
        LOG.info(f"Loading model initialization from {model_pt}")
        #--------ADDED CODE------------
        if model_class_name == "GPT2LMHeadModel":
        # Adding padding to tokenizer
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)
        #-------------------------------
        state_dict = torch.load(base_dir+model_pt, map_location="cpu")

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            LOG.info("Default load failed; stripping prefix and trying again.")
            state_dict = {re.sub("^model.", "", k): v for k,
                          v in state_dict.items()}

            model.load_state_dict(state_dict)

        LOG.info("Loaded model initialization")

    if dropout is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = dropout
                    n_reset += 1

            if hasattr(m, "activation_dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = dropout
                    n_reset += 1

        LOG.info(f"Set {n_reset} dropout modules to p={dropout}")

    param_names = [n for n, _ in model.named_parameters()]
    bad_inner_params = [p for p in model_inner_params if p not in param_names]
    if len(bad_inner_params) != 0:
        raise ValueError(
            f"Params {bad_inner_params} do not exist in model of type {type(model)}.")

    if no_grad_layers is not None:
        if half:
            model.bfloat16()

        def upcast(mod):
            modlist = None
            for child in mod.children():
                if isinstance(child, nn.ModuleList):
                    assert modlist is None, f"Found multiple modlists for {mod}"
                    modlist = child
            if modlist is None:
                raise RuntimeError("Couldn't find a ModuleList child")

            LOG.info(
                f"Setting {len(modlist) - no_grad_layers} modules to full precision, with autocasting")
            modlist[no_grad_layers:].to(torch.float32)
            modlist[no_grad_layers] = CastModule(modlist[no_grad_layers])
            modlist[-1] = CastModule(modlist[-1],
                                     in_cast=torch.float32, out_cast=torch.bfloat16)

        parents = []
        if hasattr(model, "transformer"):
            parents.append(model.transformer)
        if hasattr(model, "encoder"):
            parents.append(model.encoder)
        if hasattr(model, "decoder"):
            parents.append(model.decoder)
        if hasattr(model, "model"):
            parents.extend([model.model.encoder, model.model.decoder])

        for t in parents:
            t.no_grad_layers = no_grad_layers
            if half:
                upcast(t)

        if half:
            idxs = []
            for p in model.inner_params:
                for comp in p.split('.'):
                    if comp.isdigit():
                        idxs.append(int(comp))
            max_idx, min_idx = str(max(idxs)), str(no_grad_layers)
            for pidx, p in enumerate(model.inner_params):
                comps = p.split('.')
                if max_idx in comps or min_idx in comps:
                    index = comps.index(
                        max_idx) if max_idx in comps else comps.index(min_idx)
                    comps.insert(index + 1, 'underlying')
                    new_p = '.'.join(comps)
                    LOG.info(
                        f"Replacing config.model.inner_params[{pidx}] '{p}' -> '{new_p}'")
                    model.inner_params[pidx] = new_p
    return model


def get_tokenizer(model_tokenizer_name, model_name, model_tokenizer_class):
    tok_name = model_tokenizer_name if model_tokenizer_name is not None else model_name
    return getattr(transformers, model_tokenizer_class).from_pretrained(tok_name, cache_dir=scr())
# ---------------------------------------------------------------------------

# Creates dictionary
def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v
    return new_dict


# Code from https://github.com/xhan77/influence-function-analysis/blob/78d5a967aba885f690d34e88d68da8678aee41f1/bert_util.py#L336
def unflatten_to_param_dim(x, param_shape_tensor):
    tar_p = []
    ptr = 0
    for p in param_shape_tensor:
        len_p = torch.numel(p)
        tmp = x[ptr: ptr + len_p].view(p.shape)
        tar_p.append(tmp)
        ptr += len_p
    return tar_p

# Code from https://github.com/xhan77/influence-function-analysis/blob/78d5a967aba885f690d34e88d68da8678aee41f1/bert_util.py#L336
def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

# Average gradient of loss function
def avg_grad(data, model, device, param_influence, task):
    # Initialize correctly on the right device
    grad_all = [torch.zeros_like(v) for v in model.parameters()]
    count = 0
    for batch in data:
        loss = multi_loss_fn(model, batch, task)
        batch_grad = torch.autograd.grad(loss, param_influence)
        bs = len(batch)  # batch size
        grad_all = [(a*count)/(count+bs)+(bt*bs)/(count+bs)
                    for a, bt in zip(grad_all, batch_grad)]
        count += bs
    return grad_all

# Hessian vector product over one batch
def hvp_fn(model, loss, multiplier):
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    gnorm = 0
    for m, g in zip(multiplier, grad):
        gnorm = gnorm + torch.dot(torch.flatten(m), torch.flatten(g))
    hvp = torch.autograd.grad(gnorm, model.parameters(), create_graph=True)
    return [h.detach() for h in hvp]

# Hessian vector product average over batches
@torch.enable_grad()
def avg_hvp(data, model, multiplier, device, task='zsre', break_early=False):
    # Initialize correctly on the right device
    hvp_all = [torch.zeros_like(v) for v in model.parameters()]
    count = 0
    for i, batch in enumerate(data):
        loss = multi_loss_fn(model, batch, task)
        hvp_batch = hvp_fn(model, loss, multiplier)
        bs = len(batch)  # batch size
        hvp_all = [(a*count)/(count+bs)+(bt*bs)/(count+bs)
                   for a, bt in zip(hvp_all, hvp_batch)]
        count += bs
        if (break_early == True) & (i == len(data)//2):
            break
    return hvp_all

# Epoch loss for SGD, SVRG, and CGD
def epoch_loss(ihvp, train_loader, model, device, target, lambda1, task='zsre'):
    # .5X^T H X + X^T lambda1 X - VX
    t1 = .5 * (torch.dot(gather_flat_grad(ihvp),
               gather_flat_grad(avg_hvp(train_loader, model, ihvp, device, task=task))))
    t2 = .5 * lambda1 * \
        torch.dot(gather_flat_grad(ihvp), gather_flat_grad(ihvp))
    t3 = torch.dot(gather_flat_grad(target), gather_flat_grad(ihvp))
    loss = t1 + t2 - t3
    return (loss)

# Epoch loss for arnoldi
def arnoldi_loss(eigvecs, eigvals, target):
    # -.5 V^TU \lambda U^TV
    grad = []
    for i, g in enumerate(target):
        grad.append(g.clone().detach().to("cpu"))
    # Project gradients onto eigvecs
    proj_grad = []
    for ev in eigvecs:
        proj_grad.append(sum([torch.dot(torch.flatten(
            g.clone().detach()), torch.flatten(e)) for g, e in zip(grad, ev)]))
    # Diag of eigenvals
    inverse_hessian_diag = 1 / eigvals
    return torch.dot(gather_flat_grad(proj_grad), inverse_hessian_diag * gather_flat_grad(proj_grad))

# Create pytorch dataloader
def create_dataloader(data, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=False)
    return (dataloader)