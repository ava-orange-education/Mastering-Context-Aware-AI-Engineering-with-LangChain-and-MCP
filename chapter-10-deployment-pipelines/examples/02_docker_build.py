"""
Example 2: Docker Build and Run
Demonstrates building and running Docker containers
"""

import subprocess
import os
import sys


def run_command(command, description):
    """Run a shell command and print output"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {command}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        
        print(f"✅ {description} - Success")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False


def check_docker_installed():
    """Check if Docker is installed"""
    try:
        subprocess.run(
            ["docker", "--version"],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_main_image():
    """Build main API Docker image"""
    return run_command(
        "docker build -f deployment/docker/Dockerfile -t rag-api:latest .",
        "Building Main API Image"
    )


def build_retrieval_image():
    """Build retrieval agent image"""
    return run_command(
        "docker build -f deployment/docker/Dockerfile.retrieval -t rag-retrieval:latest .",
        "Building Retrieval Agent Image"
    )


def build_analysis_image():
    """Build analysis agent image"""
    return run_command(
        "docker build -f deployment/docker/Dockerfile.analysis -t rag-analysis:latest .",
        "Building Analysis Agent Image"
    )


def build_synthesis_image():
    """Build synthesis agent image"""
    return run_command(
        "docker build -f deployment/docker/Dockerfile.synthesis -t rag-synthesis:latest .",
        "Building Synthesis Agent Image"
    )


def list_images():
    """List Docker images"""
    return run_command(
        "docker images | grep rag",
        "Listing RAG Docker Images"
    )


def run_container():
    """Run the main API container"""
    print(f"\n{'='*70}")
    print("Running Main API Container")
    print(f"{'='*70}")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found")
        print("Create .env file with required environment variables")
        return False
    
    command = """
    docker run -d \
        --name rag-api \
        -p 8000:8000 \
        --env-file .env \
        rag-api:latest
    """
    
    return run_command(command, "Starting Container")


def stop_container():
    """Stop and remove container"""
    run_command("docker stop rag-api", "Stopping Container")
    run_command("docker rm rag-api", "Removing Container")


def view_logs():
    """View container logs"""
    return run_command(
        "docker logs rag-api",
        "Viewing Container Logs"
    )


def docker_compose_up():
    """Start all services with docker-compose"""
    return run_command(
        "docker-compose -f deployment/docker/docker-compose.yaml up -d",
        "Starting All Services with Docker Compose"
    )


def docker_compose_down():
    """Stop all services"""
    return run_command(
        "docker-compose -f deployment/docker/docker-compose.yaml down",
        "Stopping All Services"
    )


def docker_compose_logs():
    """View docker-compose logs"""
    return run_command(
        "docker-compose -f deployment/docker/docker-compose.yaml logs",
        "Viewing Docker Compose Logs"
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Docker Build and Run Examples")
    print("="*70)
    
    # Check Docker installation
    if not check_docker_installed():
        print("\n❌ Docker is not installed or not in PATH")
        print("Install Docker: https://docs.docker.com/get-docker/")
        sys.exit(1)
    
    print("\n✅ Docker is installed")
    
    # Menu
    print("\nChoose an option:")
    print("1. Build all images")
    print("2. Build main API image only")
    print("3. Run main API container")
    print("4. Stop main API container")
    print("5. View container logs")
    print("6. Start all services (docker-compose)")
    print("7. Stop all services (docker-compose)")
    print("8. View docker-compose logs")
    print("9. List RAG images")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-9): ")
    
    if choice == "1":
        build_main_image()
        build_retrieval_image()
        build_analysis_image()
        build_synthesis_image()
        list_images()
    
    elif choice == "2":
        build_main_image()
    
    elif choice == "3":
        run_container()
    
    elif choice == "4":
        stop_container()
    
    elif choice == "5":
        view_logs()
    
    elif choice == "6":
        docker_compose_up()
    
    elif choice == "7":
        docker_compose_down()
    
    elif choice == "8":
        docker_compose_logs()
    
    elif choice == "9":
        list_images()
    
    elif choice == "0":
        print("Exiting...")
    
    else:
        print("Invalid choice")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)