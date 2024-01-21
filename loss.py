import torch
import soundfile as sf
def Loss(guass, data_predict, data_orig):

    kl_loss = 0.5*(guass[:, :, 1] + 1-guass[:, :, 0]**2-torch.exp(guass[:, :, 1]))
      
    n_l2 = torch.mean(torch.pow(data_predict-data_orig, 2), dim=[1,2])
    s_l2 = torch.mean(torch.pow(data_orig, 2), dim=[1,2])
    snr = 10 * torch.log10(s_l2/n_l2)
    avg_snr = torch.mean(snr)
    avg_mse = torch.mean(torch.sqrt(n_l2))
    
    return avg_mse + torch.sum(kl_loss), avg_snr

def mlloss(data_inputs, data_outputs, data_orig):


    orig_pesq = pesq(data_inputs, data_orig)
    enhan_pesq = pesq(data_outputs, data_orig)

    reward = torch.max(torch.tensor(0.0).cuda(), enhan_pesq - orig_pesq)
    return reward













