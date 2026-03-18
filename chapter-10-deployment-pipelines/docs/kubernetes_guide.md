# Kubernetes Deployment Guide

Complete guide to deploying the RAG Multi-Agent System on Kubernetes.

## Prerequisites

- Kubernetes cluster 1.24+
- kubectl configured
- 8GB+ RAM available in cluster
- Docker registry access
- Persistent storage (optional)

## Quick Start

### 1. Prepare Environment
```bash
# Set environment variables
export DOCKER_REGISTRY=your-registry.io
export IMAGE_TAG=latest
export NAMESPACE=rag-system
```

### 2. Build and Push Images
```bash
# Build images
docker build -f deployment/docker/Dockerfile -t $DOCKER_REGISTRY/rag-api:$IMAGE_TAG .

# Push to registry
docker push $DOCKER_REGISTRY/rag-api:$IMAGE_TAG
```

### 3. Configure Secrets
```bash
# Create namespace
kubectl create namespace $NAMESPACE

# Create secrets
kubectl create secret generic rag-secrets \
  --from-literal=ANTHROPIC_API_KEY=your_key \
  --from-literal=PINECONE_API_KEY=your_key \
  --from-literal=JWT_SECRET_KEY=your_secret \
  -n $NAMESPACE
```

### 4. Deploy Application
```bash
# Apply all manifests
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/api-deployment.yaml
kubectl apply -f deployment/kubernetes/api-service.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### 5. Verify Deployment
```bash
# Check pods
kubectl get pods -n $NAMESPACE

# Check services
kubectl get svc -n $NAMESPACE

# Check logs
kubectl logs -f deployment/rag-api -n $NAMESPACE
```

## Architecture
```
                    Internet
                       │
                       ▼
                 ┌──────────┐
                 │  Ingress │
                 └─────┬────┘
                       │
                       ▼
                 ┌──────────┐
                 │ Service  │
                 └─────┬────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    ┌───────┐      ┌───────┐     ┌───────┐
    │ Pod 1 │      │ Pod 2 │     │ Pod 3 │
    │ API   │      │ API   │     │ API   │
    └───┬───┘      └───┬───┘     └───┬───┘
        │              │             │
        └──────────────┼─────────────┘
                       │
                       ▼
                 External Services
                 (Vector Store, LLM)
```

## Kubernetes Resources

### Namespace

Create isolated environment:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    name: rag-system
```
```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
```

### ConfigMap

Store non-sensitive configuration:
```bash
# View ConfigMap
kubectl get configmap rag-config -n rag-system -o yaml

# Edit ConfigMap
kubectl edit configmap rag-config -n rag-system
```

### Secrets

Store sensitive data:
```bash
# Create from literals
kubectl create secret generic rag-secrets \
  --from-literal=API_KEY=value \
  -n rag-system

# Create from file
kubectl create secret generic rag-secrets \
  --from-env-file=.env \
  -n rag-system

# View secrets (base64 encoded)
kubectl get secret rag-secrets -n rag-system -o yaml

# Decode secret
kubectl get secret rag-secrets -n rag-system -o jsonpath='{.data.ANTHROPIC_API_KEY}' | base64 -d
```

### Deployment

Main application deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    spec:
      containers:
      - name: api
        image: rag-api:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```
```bash
# Apply deployment
kubectl apply -f deployment/kubernetes/api-deployment.yaml

# Check rollout status
kubectl rollout status deployment/rag-api -n rag-system

# View deployment details
kubectl describe deployment rag-api -n rag-system
```

### Service

Expose deployment:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: rag-api
```
```bash
# Apply service
kubectl apply -f deployment/kubernetes/api-service.yaml

# Get service details
kubectl get svc rag-api -n rag-system

# Test service internally
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://rag-api.rag-system/health
```

### Ingress

External access:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-api
            port:
              number: 80
```
```bash
# Apply ingress
kubectl apply -f deployment/kubernetes/ingress.yaml

# Get ingress details
kubectl get ingress -n rag-system

# Describe ingress
kubectl describe ingress rag-ingress -n rag-system
```

## Scaling

### Manual Scaling
```bash
# Scale deployment
kubectl scale deployment rag-api --replicas=5 -n rag-system

# Verify scaling
kubectl get pods -n rag-system -l app=rag-api
```

### Horizontal Pod Autoscaler (HPA)

Automatic scaling based on metrics:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```
```bash
# Apply HPA
kubectl apply -f deployment/kubernetes/hpa.yaml

# Check HPA status
kubectl get hpa -n rag-system

# Describe HPA
kubectl describe hpa rag-api-hpa -n rag-system
```

## Health Checks

### Liveness Probe

Restart unhealthy pods:
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Readiness Probe

Remove unhealthy pods from service:
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

### Startup Probe

Allow slow-starting containers:
```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 30
```

## Updates and Rollbacks

### Rolling Update

Update deployment with zero downtime:
```bash
# Update image
kubectl set image deployment/rag-api \
  api=rag-api:v2 \
  -n rag-system

# Watch rollout
kubectl rollout status deployment/rag-api -n rag-system

# View rollout history
kubectl rollout history deployment/rag-api -n rag-system
```

### Rollback

Revert to previous version:
```bash
# Rollback to previous version
kubectl rollout undo deployment/rag-api -n rag-system

# Rollback to specific revision
kubectl rollout undo deployment/rag-api --to-revision=2 -n rag-system

# Verify rollback
kubectl rollout status deployment/rag-api -n rag-system
```

### Update Strategy

Configure in deployment:
```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

## Monitoring

### Pod Logs
```bash
# View logs from all pods
kubectl logs -l app=rag-api -n rag-system

# Follow logs
kubectl logs -f deployment/rag-api -n rag-system

# Logs from specific pod
kubectl logs rag-api-5d4f8c7b9-abc12 -n rag-system

# Previous container logs
kubectl logs rag-api-5d4f8c7b9-abc12 --previous -n rag-system
```

### Resource Usage
```bash
# Top pods
kubectl top pods -n rag-system

# Top nodes
kubectl top nodes

# Describe pod
kubectl describe pod rag-api-5d4f8c7b9-abc12 -n rag-system
```

### Events
```bash
# Get events
kubectl get events -n rag-system

# Watch events
kubectl get events -n rag-system --watch

# Sort by timestamp
kubectl get events --sort-by=.metadata.creationTimestamp -n rag-system
```

## Troubleshooting

### Pod Issues
```bash
# Check pod status
kubectl get pods -n rag-system

# Describe pod
kubectl describe pod <pod-name> -n rag-system

# Get pod logs
kubectl logs <pod-name> -n rag-system

# Execute command in pod
kubectl exec -it <pod-name> -n rag-system -- /bin/bash

# Port forward for debugging
kubectl port-forward pod/<pod-name> 8000:8000 -n rag-system
```

### Service Issues
```bash
# Check endpoints
kubectl get endpoints rag-api -n rag-system

# Test service from debug pod
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://rag-api.rag-system/health
```

### Network Issues
```bash
# Check network policies
kubectl get networkpolicies -n rag-system

# Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup rag-api.rag-system
```

### Common Problems

**Pods not starting:**
```bash
# Check events
kubectl describe pod <pod-name> -n rag-system

# Common causes:
# - Image pull errors
# - Resource constraints
# - Configuration errors
```

**ImagePullBackOff:**
```bash
# Check image name and registry
kubectl get pod <pod-name> -n rag-system -o yaml | grep image:

# Verify registry credentials
kubectl get secrets -n rag-system
```

**CrashLoopBackOff:**
```bash
# Check logs
kubectl logs <pod-name> -n rag-system

# Check previous logs
kubectl logs <pod-name> --previous -n rag-system
```

## Security

### RBAC

Create service account with limited permissions:
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rag-api-sa
  namespace: rag-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: rag-api-role
  namespace: rag-system
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: rag-api-binding
  namespace: rag-system
subjects:
- kind: ServiceAccount
  name: rag-api-sa
roleRef:
  kind: Role
  name: rag-api-role
  apiGroup: rbac.authorization.k8s.io
```

### Pod Security
```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
  - name: api
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

### Network Policies

Restrict network access:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-api-netpol
spec:
  podSelector:
    matchLabels:
      app: rag-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

## Backup and Disaster Recovery

### Backup Resources
```bash
# Backup all resources
kubectl get all -n rag-system -o yaml > backup.yaml

# Backup specific resources
kubectl get deployment,service,configmap,secret -n rag-system -o yaml > backup.yaml
```

### Restore
```bash
# Restore from backup
kubectl apply -f backup.yaml
```

## Cleanup
```bash
# Delete deployment
kubectl delete deployment rag-api -n rag-system

# Delete service
kubectl delete service rag-api -n rag-system

# Delete entire namespace (all resources)
kubectl delete namespace rag-system
```

## Production Best Practices

1. **Use resource limits**
   - Prevent resource exhaustion
   - Ensure fair resource distribution

2. **Implement health checks**
   - Liveness, readiness, startup probes
   - Automatic recovery from failures

3. **Enable autoscaling**
   - HPA for dynamic scaling
   - Match traffic patterns

4. **Use multiple replicas**
   - Minimum 3 for high availability
   - Spread across zones if possible

5. **Monitor everything**
   - Logs, metrics, traces
   - Set up alerts

6. **Plan updates carefully**
   - Rolling updates with proper strategy
   - Test in staging first

7. **Secure your cluster**
   - RBAC, network policies
   - Pod security policies
   - Regular updates