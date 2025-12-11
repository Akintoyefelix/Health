
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Verifying imports...")

try:
    from src.core import config
    print("✅ src.core.config imported")
    
    from src.core import models
    print("✅ src.core.models imported")
    
    from src.core import index
    print("✅ src.core.index imported")
    
    from src.core import engine
    print("✅ src.core.engine imported")
    
    from src.data import loader
    print("✅ src.data.loader imported")
    
    from src.utils import uniprot
    print("✅ src.utils.uniprot imported")
    
    from src import main
    print("✅ src.main imported")
    
    print("\nAll modules imported successfully!")
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    sys.exit(1)
