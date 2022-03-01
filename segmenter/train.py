import argparse
import os
import tensorflow as tf
import tensorflow_io as tfio
from segmenter import vit
import segmenter.ADE20K as data


def build_model(args, tune_config=None, hvd=None):
    if tune_config:
        # update HABANA_VISIBLE_DEVICES based on CUDA_VISIBLE_DEVICES set by ray tune
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            os.environ['HABANA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        # update args based on tune_config
        if 'pretrained' in tune_config:
            args.pretrained = tune_config['pretrained']
        if 'model' in tune_config:
            args.model = tune_config['model']
        if 'dtype' in tune_config:
            args.dtype = tune_config['dtype']
        if 'initial_lr' in tune_config:
            args.initial_lr = tune_config['initial_lr']
        if 'mask_decoder' in tune_config:
            args.mask_decoder = tune_config['mask_decoder']
        if 'optimizer' in tune_config:
            args.optimizer = tune_config['optimizer']

    if args.dtype == 'bf16' and args.device == 'HPU':
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    if args.device == 'HPU':
        from habana_frameworks.tensorflow import load_habana_module
        from habana_frameworks.tensorflow.ops.layer_norm import HabanaLayerNormalization
        from habana_frameworks.tensorflow.ops.gelu import habana_gelu
        load_habana_module()
        tf.keras.layers.LayerNormalization = HabanaLayerNormalization
        tf.keras.activations.gelu = habana_gelu

    num_classes = data.num_classes

    nb_epoch = args.epochs
    initial_lr = args.initial_lr
    final_lr = args.final_lr
    model_name = args.model
    num_decoder_layers = 2 if args.mask_decoder else 0

    image_size = data.image_size
    if model_name == 'ViT-B_16':
        model = vit.vit_b16(
            image_size=image_size,
            pretrained=args.pretrained,
            classes=num_classes,
            weights="imagenet21k",
            num_decoder_layers=num_decoder_layers)
    elif model_name == 'ViT-L_16':
        model = vit.vit_l16(
            image_size=image_size,
            pretrained=args.pretrained,
            classes=num_classes,
            weights="imagenet21k",
            num_decoder_layers=num_decoder_layers)
    elif model_name == 'ViT-B_32':
        model = vit.vit_b32(
            image_size=image_size,
            pretrained=args.pretrained,
            classes=num_classes,
            weights="imagenet21k",
            num_decoder_layers=num_decoder_layers)
    elif model_name == 'ViT-L_32':
        model = vit.vit_l32(
            image_size=image_size,
            pretrained=args.pretrained,
            classes=num_classes,
            weights="imagenet21k",
            num_decoder_layers=num_decoder_layers)
    else:
        print(
            "Model is not supported, please use either ViT-B_16 or ViT-L_16 or ViT-B_32 or ViT-L_32")
        exit(0)

    batch_size = args.batch_size
    steps_per_epoch = data.train_samples // batch_size
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch

    total_steps = nb_epoch * steps_per_epoch

    lrate = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_lr, total_steps, end_learning_rate=final_lr, power=0.9)

    if optim_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lrate, momentum=0.9)
    elif optim_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
    elif optim_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lrate, rho=0.9)
    else:
        raise ValueError

    if hvd is not None:
        optimizer = hvd.DistributedOptimizer(optimizer)
    from segmenter.layers import SparseCategoricalCrossentropyEx, SparseCategoricalAccuracyEx, SparseCategoricalMeanIoU
    loss = SparseCategoricalCrossentropyEx(from_logits=True)
    accuracy = SparseCategoricalAccuracyEx()
    iou = SparseCategoricalMeanIoU(num_classes,name='mean_iou')
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[accuracy,iou], run_eagerly=False)
    return model

def main(args, tune_config=None, hvd=None):
    model = build_model(args,tune_config,hvd)
    model_name = args.model
    batch_size = args.batch_size
    nb_epoch = args.epochs
    dataset = args.dataset
    resume_from_checkpoint_path = args.resume_from_checkpoint_path
    resume_from_epoch = args.resume_from_epoch

    steps_per_epoch = data.train_samples // batch_size
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    validation_steps = data.test_samples // batch_size
    if args.validation_steps is not None:
        validation_steps = args.validation_steps
    save_name = model_name if not model_name.endswith('.h5') else \
        os.path.split(model_name)[-1].split('.')[0].split('-')[0]

    save_name = f'{save_name}-{"pretrained-" if args.pretrained else ""}-{"mask" if args.mask_decoder else "lin"}decoder'
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
       os.path.join(args.outdir,'ckpts', save_name) + '-ckpt-{epoch:03d}.h5',
       monitor='train_loss')

    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    if hvd is None or hvd.rank() == 0:
        callbacks.append(model_ckpt)
        if args.profile:
            callbacks.append(tf.keras.callbacks.TensorBoard(os.path.join(args.outdir,'profile'),write_graph=False, profile_batch='3,5'))

    if tune_config:
        from ray.tune.integration.keras import TuneReportCallback

        callbacks.append(TuneReportCallback({"mean_accuracy": "sparse_categorical_accuracy"}))

    if hvd is not None:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    ds_train = data.get_dataset(dataset, batch_size,
                           is_training=True)
    ds_valid = data.get_dataset(dataset, batch_size,
                           is_training=False)


    if (args.predict_checkpoint_path is not None):
        # save predictions to s3
        model.load_weights(args.predict_checkpoint_path)
        import numpy as np
        import cv2
        #random colors
        colors = np.random.randint(256, size=(data.num_classes+1, 3)).astype(np.uint8)
        num_preds = 20
        ind = 0
        pred_path = os.path.join(args.outdir,'predictions')
        for sample in ds_valid:
            x,y = sample
            batch_pred = model.predict(x=x, batch_size=batch_size,steps=1)
            batch_pred = np.argmax(batch_pred,axis=-1).astype(np.uint8)
            images = ((x.numpy()/2 + 0.5)*255.).astype(np.uint8)
            for b in range(batch_size):
                ind = ind+1
                image = images[b]
                pred = batch_pred[b]
                labels = y[b]
                cv2.imwrite(os.path.join(pred_path,f'image_{ind}.jpg'),image)
                y_pred = colors[pred+1]
                y_true = colors[labels]
                cv2.imwrite(os.path.join(pred_path, f'labels_{ind}.png'), y_true)
                cv2.imwrite(os.path.join(pred_path, f'pred_{ind}.png'), y_pred)
            if ind >= num_preds:
                break
        return

    if (args.evaluate_checkpoint_path is not None):
        model.load_weights(args.evaluate_checkpoint_path)
        results = model.evaluate(x=ds_valid, steps=validation_steps)
        print("Test loss, Test acc:", results)
        return

    if ((resume_from_epoch is not None) and (resume_from_checkpoint_path is not None)):
        print(f'resume from epoch {resume_from_epoch} and from checkpoint {resume_from_checkpoint_path}')
        model.load_weights(resume_from_checkpoint_path)

    if validation_steps > 0 and hvd is None:
        model.fit(x=ds_train, y=None,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  initial_epoch=resume_from_epoch,
                  epochs=nb_epoch,
                  shuffle=True,
                  verbose=1,
                  validation_data=(ds_valid, None),
                  validation_steps=validation_steps,
                  )
    else:
        model.fit(x=ds_train, y=None,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  initial_epoch=resume_from_epoch,
                  epochs=nb_epoch,
                  shuffle=True,
                  verbose=1 if hvd is None or hvd.rank() == 0 else 0,
                  )
        if not model.stop_training and (hvd is None or hvd.rank() == 0) and validation_steps > 0:
            results = model.evaluate(x=ds_valid, steps=validation_steps)
            print("Test loss, Test acc:", results)


    if not model.stop_training and (hvd is None or hvd.rank() == 0):
        model.save(os.path.join(args.outdir,'ckpts', save_name)+'-model-final.h5')


def prepare_env(args,model=None,chief=True):
    # prepare environment
    # download data and weights
    import subprocess
    rootdir = os.environ.get("HOME","")
    if args.dataset.startswith('s3://'):
        local_data_path = f'{rootdir}/.keras/datasets/tf_records'
        if chief and not os.path.exists(local_data_path):
            cmd = f"aws s3 sync {args.dataset} {local_data_path}"
            subprocess.call(cmd.split())
        args.dataset = local_data_path
    if model is not None:
        # download pretrained encoder weights
        weights = "imagenet21k"
        fname = f"{model}_{weights}.npz"
        if chief and not os.path.exists(f'{rootdir}/.keras/weights/{fname}'):
            origin = f"https://github.com/faustomorales/vit-keras/releases/download/dl/{fname}"
            tf.keras.utils.get_file(fname, origin, cache_subdir="weights", cache_dir='~/.keras')
    if chief and not os.path.exists(os.path.join(args.outdir,'ckpts')):
        os.makedirs(os.path.join(args.outdir,'ckpts'))
    if chief and not os.path.exists(os.path.join(args.outdir,'predictions')):
        os.makedirs(os.path.join(args.outdir,'predictions'))
    if args.resume_from_checkpoint_path is not None and args.resume_from_checkpoint_path.startswith('s3://'):
        local_ckpt = f'{rootdir}/.keras/ckpt/resume.h5'
        if chief:
             cmd = f"aws s3 cp {args.resume_from_checkpoint_path} {local_ckpt}"
             subprocess.call(cmd.split())
        args.resume_from_checkpoint_path = local_ckpt
    if args.evaluate_checkpoint_path is not None and args.evaluate_checkpoint_path.startswith('s3://'):
        local_ckpt = f'{rootdir}/.keras/ckpt/eval.h5'
        if chief:
            cmd = f"aws s3 cp {args.evaluate_checkpoint_path} {local_ckpt}"
            subprocess.call(cmd.split())
        args.evaluate_checkpoint_path = local_ckpt
    if args.predict_checkpoint_path is not None and args.predict_checkpoint_path.startswith('s3://'):
        local_ckpt = f'{rootdir}/.keras/ckpt/predict.h5'
        if chief:
            cmd = f"aws s3 cp {args.predict_checkpoint_path} {local_ckpt}"
            subprocess.call(cmd.split())
        args.predict_checkpoint_path = local_ckpt
    return args

def sync_to_s3(local_path,s3_path):
    import subprocess
    cmd = f"aws s3 sync {local_path} {s3_path}"
    subprocess.call(cmd.split())

if __name__ == '__main__':
    try:
        # use pyhml (https://github.com/HabanaAI/pyhlml) to discover whether we are on DL1
        # this enables script to run seemlessy on different environments
        import pyhlml
        try:
            pyhlml.hlmlInit()
            device_count = pyhlml.hlmlDeviceGetCount()
            pyhlml.hlmlShutdown()
            isHPU = device_count > 0
        except:
            isHPU = False
    except:
        # if pyhlml not installed assume HPU
        isHPU = True

    parser = argparse.ArgumentParser(description='Segmenter training script.')
    parser.add_argument('--dataset', '--dataset_dir', metavar='PATH',
                        default='.keras/datasets/tf_records', help='Path to data.')
    parser.add_argument('--optimizer', default='sgd',
                        choices=['sgd', 'adam', 'rmsprop'], help='Optimizer.')
    parser.add_argument('-d', '--dtype', default='fp32',
                        choices=['fp32', 'bf16'], help='Data type.')
    parser.add_argument('--batch_size', type=int,
                        default=4, help='Global batch size.')
    parser.add_argument('--initial_lr', type=float,
                        default=1e-3, help='Initial learning rate.')
    parser.add_argument('--final_lr', type=float,
                        default=1e-5, help='Final learning rate.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Total number of epochs for training.')
    parser.add_argument('--steps_per_epoch', type=int,
                        help='Number of steps for training per epoch, overrides default value.')
    parser.add_argument('--validation_steps', type=int,
                        help='Number of steps for validation, overrides default value.')
    parser.add_argument('--model', default='ViT-B_16',
                        choices=['ViT-B_16', 'ViT-L_16', 'ViT-B_32', 'ViT-L_32'], help='Model.')
    parser.add_argument('--resume_from_checkpoint_path',
                        metavar='PATH', help='Path to checkpoint to start from.')
    parser.add_argument('--resume_from_epoch', metavar='EPOCH_INDEX',
                        type=int, default=0, help='Initial epoch index.')
    parser.add_argument('--evaluate_checkpoint_path', metavar='PATH',
                        help='Checkpoint path for evaluating the model')
    parser.add_argument('--predict_checkpoint_path', metavar='PATH',
                        help='Checkpoint path for making predictions')
    parser.add_argument('--tune_params', action='store_true',
                        default=False, help='Run hyper parameter tuning on a programmed search space.')
    parser.add_argument('--profile', action='store_true',
                        default=False, help='Profile HPU.')
    parser.add_argument('--mask_decoder', action='store_true',
                        default=False, help='Use transformer for decoder.')
    parser.add_argument('--pretrained', action='store_true',
                        default=False, help='Use pretrained encoder weights.')
    parser.add_argument('--outdir', metavar='PATH',
                        default='./output', help='Output path.')
    args = parser.parse_args()

    sync2s3 = args.outdir.startswith('s3://')

    if sync2s3:
        s3_path = args.outdir
        args.outdir = './output'
    args.device = 'HPU' if isHPU else 'GPU'

    if args.tune_params:
        args.distributed = False
        args.epochs = 1
        args.validation_steps = 0 # using the train metrics for tuning
        args = prepare_env(args,'ViT-B_16')
        args = prepare_env(args,'ViT-L_32')
        import ray
        from ray import tune
        from ray.tune.schedulers import AsyncHyperBandScheduler

        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration", max_t=5000, grace_period=20)
        resources_per_trial = {"cpu": 2}
        # if on HPU explicitly init ray with number of accelerators set to 8
        if args.device == 'HPU':
            ray.init(num_gpus=8)
            resources_per_trial={
                "cpu": 12,
                "gpu": 1
            }
        from functools import partial
        analysis = tune.run(
            partial(main, args),
            name="tune_segmenter",
            scheduler=sched,
            metric="mean_accuracy",
            mode="max",
            stop={
                "mean_accuracy": 0.9,
                "training_iteration": 5000
            },
            num_samples=1,
            resources_per_trial=resources_per_trial,
            config={
                "pretrained": tune.grid_search([True, False]),
                "model": tune.grid_search(['ViT-B_16', 'ViT-L_32']),
                "mask_decoder": tune.grid_search([True, False]),
            })
        print("Best hyperparameters found were: ", analysis.best_config)
    else:
        import horovod.tensorflow.keras as hvd
        hvd.init()
        print(f'Running {hvd.rank()} of {hvd.size()}')
        args = prepare_env(args,args.model if args.pretrained else None,chief=hvd.local_rank()==0)
        from mpi4py import MPI
        MPI.COMM_WORLD.Barrier()
        main(args,hvd=hvd)

    if sync2s3:
        sync_to_s3(args.outdir, s3_path)
