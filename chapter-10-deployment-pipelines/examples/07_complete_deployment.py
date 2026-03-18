"""
Example 7: Complete Deployment
End-to-end deployment demonstration
"""

import asyncio
import os
import sys
from pathlib import Path


async def check_prerequisites():
    """Check if all prerequisites are met"""
    print("\n" + "="*70)
    print("Checking Prerequisites")
    print("="*70 + "\n")
    
    issues = []
    
    # Check .env file
    if not Path(".env").exists():
        issues.append(".env file not found - copy from .env.example")
    else:
        print("✅ .env file exists")
    
    # Check required environment variables
    required_vars = [
        "ANTHROPIC_API_KEY",
        "PINECONE_API_KEY",
        "JWT_SECRET_KEY"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"{var} not set in environment")
        else:
            print(f"✅ {var} is set")
    
    # Check Docker
    import subprocess
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("Docker not installed or not in PATH")
    
    # Check kubectl
    try:
        subprocess.run(["kubectl", "version", "--client"], check=True, capture_output=True)
        print("✅ kubectl is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("kubectl not installed or not in PATH")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("\n✅ All prerequisites met!")
    return True


async def build_images():
    """Build Docker images"""
    print("\n" + "="*70)
    print("Building Docker Images")
    print("="*70 + "\n")
    
    import subprocess
    
    images = [
        ("deployment/docker/Dockerfile", "rag-api:latest"),
        ("deployment/docker/Dockerfile.retrieval", "rag-retrieval:latest"),
        ("deployment/docker/Dockerfile.analysis", "rag-analysis:latest"),
        ("deployment/docker/Dockerfile.synthesis", "rag-synthesis:latest"),
    ]
    
    for dockerfile, tag in images:
        print(f"Building {tag}...")
        try:
            subprocess.run(
                f"docker build -f {dockerfile} -t {tag} .",
                shell=True,
                check=True
            )
            print(f"✅ Built {tag}\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build {tag}: {e}\n")
            return False
    
    print("✅ All images built successfully!")
    return True


async def test_local():
    """Test deployment locally with docker-compose"""
    print("\n" + "="*70)
    print("Testing Local Deployment")
    print("="*70 + "\n")
    
    import subprocess
    
    print("Starting services with docker-compose...")
    
    try:
        subprocess.run(
            "docker-compose -f deployment/docker/docker-compose.yaml up -d",
            shell=True,
            check=True
        )
        print("✅ Services started\n")
        
        # Wait for services to be ready
        print("Waiting for services to be ready...")
        await asyncio.sleep(10)
        
        # Test health endpoint
        import aiohttp
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get("http://localhost:8000/health") as response:
                    if response.status == 200:
                        print("✅ API is healthy\n")
                        return True
                    else:
                        print(f"❌ API health check failed: {response.status}\n")
                        return False
            except Exception as e:
                print(f"❌ Cannot connect to API: {e}\n")
                return False
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}\n")
        return False


async def deploy_to_kubernetes():
    """Deploy to Kubernetes"""
    print("\n" + "="*70)
    print("Deploying to Kubernetes")
    print("="*70 + "\n")
    
    import subprocess
    
    manifests = [
        "deployment/kubernetes/namespace.yaml",
        "deployment/kubernetes/configmap.yaml",
        "deployment/kubernetes/secrets.yaml",
        "deployment/kubernetes/api-deployment.yaml",
        "deployment/kubernetes/api-service.yaml",
        "deployment/kubernetes/ingress.yaml",
    ]
    
    for manifest in manifests:
        print(f"Applying {manifest}...")
        try:
            subprocess.run(
                f"kubectl apply -f {manifest}",
                shell=True,
                check=True
            )
            print(f"✅ Applied {manifest}\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to apply {manifest}: {e}\n")
            return False
    
    print("✅ Deployed to Kubernetes!")
    
    # Wait for pods to be ready
    print("\nWaiting for pods to be ready...")
    await asyncio.sleep(5)
    
    # Check pod status
    try:
        subprocess.run(
            "kubectl get pods -n rag-system",
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError:
        pass
    
    return True


async def cleanup_local():
    """Clean up local deployment"""
    print("\n" + "="*70)
    print("Cleaning Up Local Deployment")
    print("="*70 + "\n")
    
    import subprocess
    
    try:
        subprocess.run(
            "docker-compose -f deployment/docker/docker-compose.yaml down",
            shell=True,
            check=True
        )
        print("✅ Local services stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to stop services: {e}")


async def cleanup_kubernetes():
    """Clean up Kubernetes deployment"""
    print("\n" + "="*70)
    print("Cleaning Up Kubernetes Deployment")
    print("="*70 + "\n")
    
    import subprocess
    
    confirm = input("This will delete all resources in rag-system namespace. Continue? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Cancelled")
        return
    
    try:
        subprocess.run(
            "kubectl delete namespace rag-system",
            shell=True,
            check=True
        )
        print("✅ Kubernetes resources deleted")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to delete resources: {e}")


async def main():
    """Main deployment workflow"""
    print("\n" + "="*70)
    print("Complete Deployment Workflow")
    print("="*70)
    
    # Menu
    print("\nSelect deployment option:")
    print("1. Full workflow (check → build → test local → deploy K8s)")
    print("2. Check prerequisites only")
    print("3. Build Docker images")
    print("4. Test locally (docker-compose)")
    print("5. Deploy to Kubernetes")
    print("6. Cleanup local deployment")
    print("7. Cleanup Kubernetes deployment")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-7): ")
    
    if choice == "1":
        # Full workflow
        if not await check_prerequisites():
            return
        
        if not await build_images():
            return
        
        if not await test_local():
            print("\n⚠️  Local testing failed. Fix issues before deploying to K8s.")
            return
        
        proceed = input("\nLocal testing successful. Deploy to Kubernetes? (y/n): ")
        if proceed.lower() == 'y':
            await deploy_to_kubernetes()
        
        cleanup = input("\nCleanup local deployment? (y/n): ")
        if cleanup.lower() == 'y':
            await cleanup_local()
    
    elif choice == "2":
        await check_prerequisites()
    
    elif choice == "3":
        await build_images()
    
    elif choice == "4":
        await test_local()
    
    elif choice == "5":
        await deploy_to_kubernetes()
    
    elif choice == "6":
        await cleanup_local()
    
    elif choice == "7":
        await cleanup_kubernetes()
    
    elif choice == "0":
        print("Exiting...")
    
    else:
        print("Invalid choice")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())