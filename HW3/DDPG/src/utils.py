def soft_update(online, target, tau):
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.copy_(tau * online_param.data + (1. - tau) * target_param.data)