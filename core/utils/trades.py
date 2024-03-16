import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from core.metrics import accuracy


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def LORE_v1(reg_type, logits_adv, logits, y, classes):

    correct = 0
    wrong = 0
    pred = torch.max(logits, dim=1)[1]
    probs = F.softmax(logits, dim=1)

    pred_adv = torch.max(logits_adv, dim=1)[1]
    probs_adv = F.softmax(logits_adv, dim=1)

    if reg_type == 'kl':
        fluc_logit_correct_prob = torch.tensor(0, dtype=torch.float32)
        fluc_logit_correct_prob = fluc_logit_correct_prob.cuda()
        fluc_logit_wrong_prob = torch.tensor(0, dtype=torch.float32)
        fluc_logit_wrong_prob = fluc_logit_wrong_prob.cuda()

        for idx_l in range(y.size(0)):
            logits_softmax = torch.softmax(logits[idx_l], dim=0)
            logits_softmax_max = torch.tensor(0, dtype=torch.float32)
            
            for j in range(len(logits_softmax)):
                if j != y[idx_l].cpu().numpy():
                    logits_softmax_max = torch.max(logits_softmax[j], logits_softmax_max)
            
            if pred[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
                correct += 1
                fluc_logit_correct_prob -= torch.log(1 - logits_softmax_max)

            else:
                wrong += 1
                fluc_logit_wrong_prob -= torch.log(1 - torch.abs(probs_adv[idx_l][pred_adv[idx_l]] - probs[idx_l][pred_adv[idx_l]]))

        fluc_logit_correct_prob /= correct
        fluc_logit_wrong_prob /= wrong
    
        return fluc_logit_correct_prob, fluc_logit_wrong_prob

    elif reg_type == 'mse':
        fluc_logit_correct = torch.tensor(0, dtype=torch.float32)
        fluc_logit_correct = fluc_logit_correct.cuda()
        fluc_logit_wrong = torch.tensor(0, dtype=torch.float32)
        fluc_logit_wrong = fluc_logit_wrong.cuda()
        
        for idx_l in range(y.size(0)):
            logits_softmax = torch.softmax(logits[idx_l], dim=0)

            if pred[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
                correct += 1
                mean_softmax = ((1 - logits_softmax[y[idx_l].cpu().numpy()])/(classes-1))
                for idx in range(len(logits_softmax)):
                    if idx != y[idx_l].cpu().numpy():
                        fluc_logit_correct += ((logits_softmax[idx] - mean_softmax)**2)
            else:
                wrong += 1
                logits_adv_softmax = torch.softmax(logits_adv[idx_l], dim=0)
                fluc_logit_wrong += (torch.sum((logits_adv_softmax - logits_softmax)**2))
        
        fluc_logit_correct /= correct
        fluc_logit_wrong /= wrong
        return fluc_logit_correct, fluc_logit_wrong


def LORE(reg_type, logits_adv, logits, y, classes, choice='wrong'):

    correct = 0
    wrong = 0

    pred = torch.max(logits, dim=1)[1]
    pred_adv = torch.max(logits_adv, dim=1)[1]

    probs = F.softmax(logits, dim=1)
    probs_adv = F.softmax(logits_adv, dim=1)

    if reg_type == 'kl':
        fluc_logit_correct_prob = torch.tensor(0, dtype=torch.float32)
        fluc_logit_correct_prob = fluc_logit_correct_prob.cuda()
        fluc_logit_wrong_prob = torch.tensor(0, dtype=torch.float32)
        fluc_logit_wrong_prob = fluc_logit_wrong_prob.cuda()
        
        alp_prob = torch.tensor(0, dtype=torch.float32)
        alp_prob = alp_prob.cuda()

        for idx_l in range(y.size(0)):
            logits_softmax = torch.softmax(logits[idx_l], dim=0)
            logits_softmax_max = torch.tensor(0, dtype=torch.float32)
            
            for j in range(len(logits_softmax)):
                if j != y[idx_l].cpu().numpy():
                    logits_softmax_max = torch.max(logits_softmax[j], logits_softmax_max)
            
            if pred[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
                correct += 1
                fluc_logit_correct_prob -= torch.log(1 - logits_softmax_max)
                if choice == 'correct':
                    alp_prob -= torch.log(1 - torch.abs(probs_adv[idx_l][pred_adv[idx_l]] - probs[idx_l][pred_adv[idx_l]]))
            else:
                wrong += 1
                fluc_logit_wrong_prob -= torch.log(1 - logits_softmax_max)
                if choice == 'wrong':
                    alp_prob -= torch.log(1 - torch.abs(probs_adv[idx_l][pred_adv[idx_l]] - probs[idx_l][pred_adv[idx_l]]))

        fluc_logit_correct_prob /= correct
        fluc_logit_wrong_prob /= wrong
        if choice == 'correct':
            alp_prob /= correct
        elif choice == 'wrong':
            alp_prob /= wrong

        return fluc_logit_correct_prob, fluc_logit_wrong_prob, alp_prob

    elif reg_type == 'mse':
        fluc_logit_correct = torch.tensor(0, dtype=torch.float32)
        fluc_logit_correct = fluc_logit_correct.cuda()
        fluc_logit_wrong = torch.tensor(0, dtype=torch.float32)
        fluc_logit_wrong = fluc_logit_wrong.cuda()

        alp = torch.tensor(0, dtype=torch.float32)
        alp = alp.cuda()

        for idx_l in range(y.size(0)):
            logits_softmax = torch.softmax(logits[idx_l], dim=0)
            mean_softmax = ((1 - logits_softmax[y[idx_l].cpu().numpy()])/(classes-1))
            logits_adv_softmax = torch.softmax(logits_adv[idx_l], dim=0)

            if pred[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
                correct += 1
                for idx in range(len(logits_softmax)):
                    if idx != y[idx_l].cpu().numpy():
                        fluc_logit_correct += ((logits_softmax[idx] - mean_softmax)**2)
                if choice == 'correct':
                    alp += (torch.sum((logits_adv_softmax - logits_softmax)**2))

            else:
                wrong += 1
                for idx in range(len(logits_softmax)):
                    if idx != y[idx_l].cpu().numpy():
                        fluc_logit_wrong += ((logits_softmax[idx] - mean_softmax)**2)
                if choice == 'wrong':
                    alp += (torch.sum((logits_adv_softmax - logits_softmax)**2))
        
        fluc_logit_correct /= correct
        fluc_logit_wrong /= wrong
        if choice == 'correct':
            alp /= correct
        elif choice == 'wrong':
            alp /= wrong

        return fluc_logit_correct, fluc_logit_wrong, alp


def trades_loss(model, x_natural, y, epoch, LORE_type, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, 
                attack='linf-pgd', num_classes=10):
    """
    TRADES training (Zhang et al, 2019).
    """
  
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    p_natural = F.softmax(model(x_natural), dim=1)
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward(retain_graph=True)
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_natural, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust

    if LORE_type != 'None':
        if epoch <= 1:
            if LORE_type == 'LORE':
                fluc_correct, fluc_wrong, alp = LORE('mse', logits_adv, logits_natural, y, num_classes, choice='wrong')
                reg = fluc_correct - fluc_wrong
                loss += (reg + 0.05 * alp)
            elif LORE_type == 'LORE_v1':
                fluc_correct, fluc_wrong = LORE_v1('mse', logits_adv, logits_natural, y, num_classes)
                reg = fluc_correct - fluc_wrong
                loss += reg
        elif epoch >= 90:
            if LORE_type == 'LORE':
                fluc_correct, fluc_wrong, alp = LORE('mse', logits_adv, logits_natural, y, num_classes, choice='correct')
                reg = fluc_wrong - fluc_correct            
                loss += (reg + 0.05 * alp)
            elif LORE_type == 'LORE_v1':
                fluc_correct, fluc_wrong = LORE_v1('mse', logits_adv, logits_natural, y, num_classes)
                reg = fluc_wrong - fluc_correct
                loss += reg
        else:
            pass
    elif LORE_type == 'None':
        pass

    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics

def trades_loss_LSE(model, x_natural, y, epoch, LORE_type, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, 
                attack='linf-pgd', label_smoothing=0.1, num_classes=10):
    """
    SCORE training (Ours).
    """
    # criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    # criterion_kl = nn.KLDivLoss(reduction='sum')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    p_natural = F.softmax(model(x_natural), dim=1)
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            output_adv = F.softmax(model(x_adv), dim=1)
            with torch.enable_grad():
                loss_lse = torch.sum((output_adv - p_natural) ** 2)
            grad = torch.autograd.grad(loss_lse, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)
  
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()
    # calculate robust loss
    y_onehot = (1 - 10 * label_smoothing / 9) * F.one_hot(y, num_classes=10) + label_smoothing / 9
    logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv = F.softmax(model(x_adv), dim=1)
    loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1)
    loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1)
    loss = loss_natural.mean() + beta * loss_robust.mean()

    if LORE_type != 'None':
        if epoch <= 1:
            if LORE_type == 'LORE':
                fluc_correct, fluc_wrong, alp = LORE('mse', logits_adv, logits_natural, y, num_classes, choice='wrong')
                reg = fluc_correct - fluc_wrong
                loss += (reg + 0.05 * alp)
            elif LORE_type == 'LORE_v1':
                fluc_correct, fluc_wrong = LORE_v1('mse', logits_adv, logits_natural, y, num_classes)
                reg = fluc_correct - fluc_wrong
                loss += reg
        elif epoch >= 90:
            if LORE_type == 'LORE':
                fluc_correct, fluc_wrong, alp = LORE('mse', logits_adv, logits_natural, y, num_classes, choice='correct')
                reg = fluc_wrong - fluc_correct            
                loss += (reg + 0.05 * alp)
            elif LORE_type == 'LORE_v1':
                fluc_correct, fluc_wrong = LORE_v1('mse', logits_adv, logits_natural, y, num_classes)
                reg = fluc_wrong - fluc_correct
                loss += reg
        else:
            pass
    elif LORE_type == 'None':
        pass

    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics
