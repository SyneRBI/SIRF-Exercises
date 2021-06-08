import os

def exercises_data_path(*data_type):
    '''
    Returns the path to data used by SIRF-exercises.

    data_type: either 'PET', 'MR' or 'Synergistic', or use multiple arguments for
    subdirectories like exercises_data_path('PET', 'mMR', 'NEMA_IQ').
    '''
    try:
        # from installer?
        from .data_path import data_path
    except ImportError:
        # from ENV variable?
        data_path = os.environ.get('SIRF_EXERCISES_DATA_PATH')

    if data_path is None or not os.path.exists(data_path):
        raise RuntimeError(
            "Exercises data weren't found. Please run download_data.sh in the "
            "scripts directory")

    # # make data_type a tuple froming a subdirectory path
    # if data_type is None:
    #     data_type = ()
    # elif isinstance(data_type, str):
    #     data_type = (data_type, )

    return os.path.join(data_path, *data_type)