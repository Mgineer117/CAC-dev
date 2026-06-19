from policy.layers.sd_lqr_networks import SDCLearner
from trainer.offline_trainer import SDCTrainer

def get_SDC(env, args, logger, writer, get_f_and_B, init_epochs):
    

    # C3M needs the SDC decomposition network too when CMG pretraining is enabled.
    needs_sdc = args.algo_name in ("sd-lqr", "sd-lqr-approx", "corl") or (
        args.algo_name == "c3m" and getattr(args, "c3m_pretrain_cmg", False)
    )
    if needs_sdc:
        SDC_func = SDCLearner(
            x_dim=env.num_dim_x,
            a_dim=args.action_dim,
            hidden_dim=args.sdc_dim,
            get_f_and_B=get_f_and_B,
            nupdates=args.sdc_epochs,
            device=args.device,
        )

        SDC_trainer = SDCTrainer(
            env=env,
            SDC_func=SDC_func,
            logger=logger,
            writer=writer,
            buffer_size=args.sdc_buffer_size,
            init_epochs=init_epochs,
            epochs=args.sdc_epochs,
        )
        print("[INFO] Learning the SDC decomposition network.")
        SDC_trainer.train()
        init_epochs = init_epochs + args.sdc_epochs
    else:
        SDC_func = None

    return SDC_func, init_epochs