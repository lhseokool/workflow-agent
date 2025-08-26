"""
Simple .env file creator
Creates .env file with OpenAI API key
"""

import os


def create_env_file():
    """Create .env file with API key"""
    print("🔑 Create .env file for OpenAI API key")
    print("="*40)
    
    # Check if .env already exists
    if os.path.exists(".env"):
        print("⚠️ .env file already exists!")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("❌ Cancelled")
            return
    
    # Get API key from user
    api_key = input("\nEnter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    # Create .env file
    try:
        with open(".env", "w") as f:
            f.write(f"# OpenAI API Configuration\n")
            f.write(f"OPENAI_API_KEY={api_key}\n")
            f.write(f"\n# Optional: Default model\n")
            f.write(f"DEFAULT_MODEL=gpt-4o-mini\n")
        
        print("✅ .env file created successfully!")
        print("📁 File location: ./.env")
        print("🚀 You can now run: python workflow_agent.py")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")


def main():
    """Main function"""
    try:
        create_env_file()
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
