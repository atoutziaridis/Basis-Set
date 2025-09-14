"""
Virtual Environment Setup script for BSV GitHub Repository Prioritizer
"""

import subprocess
import sys
import os
from pathlib import Path

def create_virtual_environment():
    """Create and setup virtual environment"""
    venv_path = Path(__file__).parent / "venv"
    
    print("Creating virtual environment...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
        print("✅ Virtual environment created!")
        
        # Determine the correct python and pip paths
        if os.name == 'nt':  # Windows
            python_path = venv_path / "Scripts" / "python.exe"
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            python_path = venv_path / "bin" / "python"
            pip_path = venv_path / "bin" / "pip"
        
        return python_path, pip_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return None, None

def install_requirements(pip_path):
    """Install required packages in virtual environment"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    print("Installing required packages in virtual environment...")
    try:
        subprocess.check_call([str(pip_path), "install", "-r", str(requirements_file)])
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

def create_activation_script():
    """Create easy activation script"""
    activate_script_content = '''#!/bin/bash
# BSV Repository Prioritizer - Virtual Environment Activation

echo "Activating BSV Repository Prioritizer virtual environment..."

# Activate virtual environment
source venv/bin/activate

# Check if activation was successful
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated successfully!"
    echo "💡 Virtual environment: $VIRTUAL_ENV"
    echo ""
    echo "Available commands:"
    echo "  python src/test_collection.py          # Test data collection"
    echo "  python src/data_collection_runner.py   # Run full data collection"
    echo "  deactivate                             # Exit virtual environment"
    echo ""
else
    echo "❌ Failed to activate virtual environment"
fi
'''
    
    activate_script_path = Path(__file__).parent / "activate_env.sh"
    with open(activate_script_path, 'w') as f:
        f.write(activate_script_content)
    
    # Make it executable
    os.chmod(activate_script_path, 0o755)
    print("✅ Created activation script: activate_env.sh")

def main():
    """Main setup function"""
    print("BSV GitHub Repository Prioritizer - Virtual Environment Setup")
    print("=" * 60)
    
    success = True
    
    # Create virtual environment
    python_path, pip_path = create_virtual_environment()
    if not python_path or not pip_path:
        success = False
    
    # Install requirements
    if success and not install_requirements(pip_path):
        success = False
    
    # Setup environment
    if not setup_environment():
        success = False
    
    # Create directories
    directories = ["data", "output", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print(f"✅ Created directories: {', '.join(directories)}")
    
    # Create activation script
    create_activation_script()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your GitHub token to the .env file:")
        print("   GITHUB_TOKEN=your_token_here")
        print("2. Activate the environment: source activate_env.sh")
        print("3. Test the setup: python src/test_collection.py")
        print("4. Run data collection: python src/data_collection_runner.py")
        print("\nAlternative activation (manual):")
        print("   source venv/bin/activate")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()