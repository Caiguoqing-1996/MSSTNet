
'''
Usage:


'''


def model_prepare(model_sel, model_para):
    '''


    :param model_sel:

        all model: EEGNet: " EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces",
                   DeepNet:
    :param model_para:
    :return:
    '''
    global nChan
    if model_sel in ['EEGNet']:
        from compared_model.EEGNet import EEGNet, Loss_func
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = EEGNet(nChan=nChan, nTime=nTime, nClass=nClass)
        loss_func = Loss_func()

    elif model_sel in ['ShallowNet']:
        from compared_model.ShallowConvNet import ShallowConvNet, Loss_func
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = ShallowConvNet(num_classes=nClass, chans=nChan, samples=nTime)
        loss_func = Loss_func()


    elif model_sel in ['EEGConformer']:
        from compared_model.EEGConformer import Conformer, Loss_func
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        drop_p1 = model_para['dropout1']
        drop_p2 = model_para['dropout2']
        depth = model_para['depth']

        model = Conformer(nChan=nChan, nTime=nTime, nClass=nClass,
                          drop_p1=drop_p1, drop_p2=drop_p2, emb_size=40, depth=depth)
        loss_func = Loss_func()

    elif model_sel in ['FBCNet']:
        from compared_model.FBCNet import FBCNet, Loss_func

        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        nBands = model_para['band']
        spa_filter = model_para['spa_filter']
        model = FBCNet(nChan=nChan, nTime=nTime, nClass=nClass, nBands=nBands, m=spa_filter)
        loss_func = Loss_func()

    elif model_sel in ['IFNet']:
        from compared_model.IFNet import IFNet, Loss_func
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        nBands = model_para['band']
        model = IFNet(in_planes=nChan, out_planes=64, kernel_size=63, radix=nBands,
                      patch_size=125, time_points=nTime, num_classes=nClass)
        loss_func = Loss_func()

    elif model_sel in ['IFNetV2']:
        from compared_model.IFNet import IFNetV2, Loss_func
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = IFNetV2(in_planes=nChan, out_planes=64, kernel_size=63, radix=2,
                       patch_size=125, time_points=nTime, num_classes=nClass)
        loss_func = Loss_func()


    elif model_sel in ['LightConvNet']:
        from compared_model.LightConvNet import LightConvNet, Loss_func
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        nBands = model_para['band']
        model = LightConvNet(num_classes=nClass, num_samples=nTime, num_channels=nChan, num_bands=nBands)
        loss_func = Loss_func()

    elif model_sel in ['TSFCNet']:
        from compared_model.TSFCNet import TSFCNet4a, Loss_func
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        nBands = model_para['band']
        model = TSFCNet4a(nChan=nChan, nTime=nTime, nClass=nClass, nBands=nBands)
        loss_func = Loss_func()


    elif model_sel in ['MSVTNet']:
        from compared_model.MSVTNet import MSVTNet
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']
        model = MSVTNet(nCh=nChan, nTime=nTime, cls=nClass)


    elif model_sel in ['MSSTNet']:
        from MSSTNet import MSSTNet, Loss_func
        nChan = model_para['channel']
        nTime = model_para['time']
        nClass = model_para['class']

        model = MSSTNet(n_chan=nChan, n_time=nTime, num_classes=nClass, para=model_para)
        loss_func = Loss_func(para=model_para)



    return model, loss_func






if __name__ == "__main__":
    run_code = 0
