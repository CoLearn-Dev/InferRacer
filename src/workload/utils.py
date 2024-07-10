def cache(root_dir="tmp/"):
    """
    Cache the return value of a **method function** in disk using pickle.
    The first argument of the function must be `self`.
    If the file does not exist, call the function and store the return value in the file named `{class_name}_{func_name}_{args}_{kwargs}` in `root_dir`.
    if `enable` is False, the function will not be cached.
    Raise error if the `root_dir` does not exist.
    """
    import pickle
    import os
    import functools

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not os.path.exists(root_dir) and root_dir != "":
                raise FileNotFoundError(f"Cache root dir {root_dir} does not exist.")
            cache_path = os.path.join(
                root_dir,
                f"{args[0].__class__.__name__}_{f.__name__}_{args[1:]}_{kwargs}",
            )
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as fi:
                    return pickle.load(fi)
            else:
                ret = f(*args, **kwargs)
                with open(cache_path, "wb") as fi:
                    pickle.dump(ret, fi)
                return ret

        return wrapper

    return decorator
