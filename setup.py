"""
Setup script for BSV GitHub Repository Prioritizer
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def setup_environment():
    """Setup environment file"""
    env_template = Path(__file__).parent / ".env.template"
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists() and env_template.exists():
        print("Setting up environment file...")
        with open(env_template, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Environment file created (.env)")
        print("⚠️  Please add your GitHub token to the .env file before running data collection")
    
    return True

def main():
    """Main setup function"""
    print("BSV GitHub Repository Prioritizer - Setup")
    print("=" * 50)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Setup environment
    if not setup_environment():
        success = False
    
    # Create directories
    directories = ["data", "output", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print(f"✅ Created directories: {', '.join(directories)}")
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your GitHub token to the .env file:")
        print("   GITHUB_TOKEN=your_token_here")
        print("2. Test the setup: python3 src/test_collection.py")
        print("3. Run data collection: python3 src/data_collection_runner.py")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()