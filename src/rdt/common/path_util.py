import os, os.path as osp


def get_rdt_src() -> str:
    return os.environ['RDT_SOURCE_DIR']


def get_rdt_config() -> str:
    return osp.join(get_rdt_src(), 'config')


def get_rdt_share() -> str:
    return osp.join(get_rdt_src(), 'share')


def get_rdt_data() -> str:
    return osp.join(get_rdt_src(), 'data')


def get_rdt_recon_data() -> str:
    return osp.join(get_rdt_src(), 'data_gen/data')


def get_rdt_eval_data() -> str:
    return osp.join(get_rdt_src(), 'eval_data')


def get_rdt_descriptions() -> str:
    return osp.join(get_rdt_src(), 'descriptions')


def get_rdt_obj_descriptions() -> str:
    return osp.join(get_rdt_descriptions(), 'objects')


def get_rdt_demo_obj_descriptions() -> str:
    return osp.join(get_rdt_descriptions(), 'demo_objects')


def get_rdt_assets() -> str:
    return osp.join(get_rdt_src(), 'assets')


def get_rdt_model_weights() -> str:
    return osp.join(get_rdt_src(), 'model_weights')


def get_train_config_dir() -> str:
    return osp.join(get_rdt_config(), 'train_cfgs')


def get_eval_config_dir() -> str:
    return osp.join(get_rdt_config(), 'full_eval_cfgs')


def get_demo_config_dir() -> str:
    return osp.join(get_rdt_config(), 'full_demo_cfgs')

