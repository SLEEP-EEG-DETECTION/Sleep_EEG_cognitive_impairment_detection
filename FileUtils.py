import os
from typing import List

class FileUtils:
    """
    Check if the given directory exists. If it doesn't, create it.
    
    Args:
        path (str): The path of the directory to check or create.
        
    Returns:
        None
    """
    @staticmethod
    def __check_dir(path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def check_dir(path: str) -> bool:
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            return True
        except:
            return False
    
    @staticmethod
    def get_file_list(dir_path: str, suffix: str) -> List[str]:
        """
        Get a list of file paths in a directory and its subdirectories, filtered by suffix.

        Args:
            dir_path (str): The path to the directory.
            suffix (str): The suffix of the files to be included in the list.

        Returns:
            List[str]: A list of file paths.
        """
        file_list = []  # type: List[str]
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(suffix):
                    file_list.append(os.path.join(root, file))
        return file_list
    
    @staticmethod
    def get_file_dic(dir_path: str, suffix: str) -> dict:
        """
        Get a dictionary of file paths in a directory and its subdirectories, filtered by suffix.

        Args:
            dir_path (str): The path to the directory.
            suffix (str): The suffix of the files to be included in the dictionary.

        Returns:
            dict: A dictionary where key is file name and value is file path.
        """
        file_dic = {}  # type: dict
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(suffix):
                    file_dic[file] = os.path.join(root, file)
        return file_dic
    
