class Error(BaseException):
    """Base Class for Other Error"""
    pass


class NoModelError(Error):
    """When Given Model is not available"""
    pass


class NoModelName(Error):
    """When new model name does not exist"""
    pass


class ModelMissmatchError(Error):
    """When Given Model does not match with current weights"""
    pass


class NoWeightsAvailable(Error):
    """When there is no weights exist in givent path"""
    pass


class NoDirExist(Error):
    """When there is no dir exist"""
    pass


class NoSavedModelExist(Error):
    """When there is no model dir exist"""

    def __init__(self, path: str = None):
        if path is None:
            self.error_message = "None returned as module path"
        else:
            self.error_message = f'There is no module at given path at: {path}'

    def __str__(self):
        return self.error_message

    pass


class NoDataReader(Error):
    """When there is no data reader for test and training"""
    pass
