# Google Cloud Marketplace & Partner Publishing Guide

This directory contains solution packaging and marketplace manifests for publishing **XLAxiom** to the **Google Cloud Marketplace** and registering it as a native **Vertex AI Workbench Extension**.

## Packaging Assets
- `solution_definition.yaml`: Commercial solution specification conforming to Google Cloud Marketplace publisher standards.
- `vertex_workbench_extension.json`: Native Vertex AI Workbench / JupyterLab launcher descriptor.

## Submission Workflow
1. **Google Cloud Partner Advantage**:
   - Register or link organization in [Google Cloud Partner Portal](https://cloud.google.com/partners).
2. **Container Artifact Ingestion**:
   - Build and sign the multi-arch container image using Cloud Build:
     ```bash
     gcloud builds submit --config=cloudbuild.yaml
     ```
   - Publish to the certified marketplace container registry:
     ```bash
     docker tag xlaxiom/runner:2.2.0 us-docker.pkg.dev/google-cloud-marketplace/xlaxiom/runner:2.2.0
     docker push us-docker.pkg.dev/google-cloud-marketplace/xlaxiom/runner:2.2.0
     ```
3. **Marketplace Listing Review**:
   - Upload `solution_definition.yaml` in Marketplace Producer Portal and submit for Google technical and security review.

