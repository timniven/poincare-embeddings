"""Convenient interface for loading and saving pickles."""
import pickle
import os


def exists(pkl_dir, pkl_name):
    """Check if a pickle exists.

    Args:
      pkl_dir: String, the directory in which to save the pickle.
      pkl_name: String, the file_name for the pickle.

    Returns:
      Boolean.
    """
    return os.path.exists(os.path.join(pkl_dir, pkl_name))


def load(pkl_dir, pkl_name):
    """Load a pickle.

    Args:
      pkl_dir: String, the directory in which to save the pickle.
      pkl_name: String, the file_name for the pickle.

    Returns:
      Object.

    Raises:
      Exception if pickle not found.
    """
    file_path = os.path.join(pkl_dir, pkl_name)
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            return obj
    except FileNotFoundError:
        raise Exception('Pickle not found: %s' % file_path)


def save(obj, pkl_dir, pkl_name):
    """Save a pickle.

    Args:
      obj: Object, the object to pickle.
      pkl_dir: String, the directory in which to save the pickle.
      pkl_name: String, the file_name for the pickle.
    """
    file_path = os.path.join(pkl_dir, pkl_name)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


class Pickler:
    """Wraps pickling functions and stores root directory."""

    def __init__(self, pkl_dir):
        """Create a new Pickler.

        Args:
          pkl_dir: String. The place to store the pickles. I assume just one for
            now.
        """
        self.pkl_dir = pkl_dir

    def exists(self, name):
        """Check if a pickle exists.

        Args:
          name: String. Should NOT have .pkl at the end.

        Returns:
          Bool.
        """
        return exists(self.pkl_dir, name + '.pkl')

    def file_path(self, name):
        """Get the file path to a pickle given by name.

        Args:
          name: String. Should NOT have .pkl at the end.

        Returns:
          String.
        """
        return os.path.join(self.pkl_dir, name + '.pkl')

    def load(self, name):
        """Load a pickle.

        Args:
          name: String. Should NOT have .pkl at the end.

        Returns:
          Object.

        Raises:
          Exception if pickle not found.
        """
        return load(self.pkl_dir, name + '.pkl')

    def save(self, obj, name):
        """Save a pickle.

        Args:
          obj: Object.
          name: String. Should NOT have .pkl at the end.
        """
        save(obj, self.pkl_dir, name + '.pkl')
