# Deployments

A Deployment manages a set of Pods to run an application workload. The Deployment controller changes the actual state toward the desired state at a controlled rate.

## Updating a Deployment

Change the Pod template to trigger a rollout:

```bash
kubectl set image deployment/nginx nginx=nginx:1.25
kubectl rollout status deployment/nginx
```

## Rolling back a Deployment

Inspect rollout history before undoing a rollout:

```bash
kubectl rollout history deployment/nginx
kubectl rollout undo deployment/nginx
```

The rollback creates a new rollout from an earlier Deployment revision. Revision history is limited by `revisionHistoryLimit`.
