"""
Example 3: Kubernetes Deployment
Demonstrates deploying to Kubernetes cluster
"""

import subprocess
import sys


def run_kubectl(command, description):
    """Run kubectl command"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: kubectl {command}\n")
    
    try:
        result = subprocess.run(
            f"kubectl {command}",
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


def check_kubectl_installed():
    """Check if kubectl is installed"""
    try:
        subprocess.run(
            ["kubectl", "version", "--client"],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_namespace():
    """Create Kubernetes namespace"""
    return run_kubectl(
        "apply -f deployment/kubernetes/namespace.yaml",
        "Creating Namespace"
    )


def create_configmap():
    """Create ConfigMap"""
    return run_kubectl(
        "apply -f deployment/kubernetes/configmap.yaml",
        "Creating ConfigMap"
    )


def create_secrets():
    """Create Secrets"""
    print(f"\n{'='*70}")
    print("Creating Secrets")
    print(f"{'='*70}")
    print("\n⚠️  Warning: Update secrets with actual values before deploying!")
    print("Use: kubectl create secret generic rag-secrets -n rag-system \\")
    print("  --from-literal=ANTHROPIC_API_KEY=your-key \\")
    print("  --from-literal=PINECONE_API_KEY=your-key \\")
    print("  --from-literal=JWT_SECRET_KEY=your-key")
    
    # For demo, apply the template (which contains placeholder values)
    return run_kubectl(
        "apply -f deployment/kubernetes/secrets.yaml",
        "Applying Secrets Template"
    )


def deploy_api():
    """Deploy API"""
    return run_kubectl(
        "apply -f deployment/kubernetes/api-deployment.yaml",
        "Deploying API"
    )


def create_service():
    """Create Service"""
    return run_kubectl(
        "apply -f deployment/kubernetes/api-service.yaml",
        "Creating Service"
    )


def create_ingress():
    """Create Ingress"""
    return run_kubectl(
        "apply -f deployment/kubernetes/ingress.yaml",
        "Creating Ingress"
    )


def create_hpa():
    """Create Horizontal Pod Autoscaler"""
    return run_kubectl(
        "apply -f deployment/kubernetes/hpa.yaml",
        "Creating HPA"
    )


def get_pods():
    """Get pods in namespace"""
    return run_kubectl(
        "get pods -n rag-system",
        "Getting Pods"
    )


def get_services():
    """Get services"""
    return run_kubectl(
        "get services -n rag-system",
        "Getting Services"
    )


def get_deployments():
    """Get deployments"""
    return run_kubectl(
        "get deployments -n rag-system",
        "Getting Deployments"
    )


def describe_pod():
    """Describe a pod"""
    pod_name = input("Enter pod name: ")
    return run_kubectl(
        f"describe pod {pod_name} -n rag-system",
        f"Describing Pod {pod_name}"
    )


def get_logs():
    """Get logs from a pod"""
    pod_name = input("Enter pod name: ")
    return run_kubectl(
        f"logs {pod_name} -n rag-system --tail=50",
        f"Getting Logs from {pod_name}"
    )


def delete_deployment():
    """Delete deployment"""
    return run_kubectl(
        "delete -f deployment/kubernetes/api-deployment.yaml",
        "Deleting API Deployment"
    )


def delete_all():
    """Delete all resources"""
    return run_kubectl(
        "delete namespace rag-system",
        "Deleting Namespace (This will delete all resources)"
    )


def full_deployment():
    """Deploy everything"""
    print("\n" + "="*70)
    print("Full Deployment")
    print("="*70)
    
    create_namespace()
    create_configmap()
    create_secrets()
    deploy_api()
    create_service()
    create_ingress()
    
    print("\n" + "="*70)
    print("Deployment Summary")
    print("="*70)
    
    get_deployments()
    get_services()
    get_pods()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Kubernetes Deployment Examples")
    print("="*70)
    
    # Check kubectl installation
    if not check_kubectl_installed():
        print("\n❌ kubectl is not installed or not in PATH")
        print("Install kubectl: https://kubernetes.io/docs/tasks/tools/")
        sys.exit(1)
    
    print("\n✅ kubectl is installed")
    
    # Menu
    print("\nChoose an option:")
    print("1. Full deployment (namespace, config, secrets, deployment, service, ingress)")
    print("2. Create namespace")
    print("3. Create ConfigMap")
    print("4. Create Secrets")
    print("5. Deploy API")
    print("6. Create Service")
    print("7. Create Ingress")
    print("8. Get Pods")
    print("9. Get Services")
    print("10. Get Deployments")
    print("11. Describe Pod")
    print("12. Get Logs")
    print("13. Delete Deployment")
    print("14. Delete All Resources")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-14): ")
    
    if choice == "1":
        full_deployment()
    elif choice == "2":
        create_namespace()
    elif choice == "3":
        create_configmap()
    elif choice == "4":
        create_secrets()
    elif choice == "5":
        deploy_api()
    elif choice == "6":
        create_service()
    elif choice == "7":
        create_ingress()
    elif choice == "8":
        get_pods()
    elif choice == "9":
        get_services()
    elif choice == "10":
        get_deployments()
    elif choice == "11":
        describe_pod()
    elif choice == "12":
        get_logs()
    elif choice == "13":
        delete_deployment()
    elif choice == "14":
        confirm = input("Are you sure? This will delete ALL resources (y/n): ")
        if confirm.lower() == 'y':
            delete_all()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)