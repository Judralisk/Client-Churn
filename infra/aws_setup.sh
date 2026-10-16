#!/usr/bin/env bash
# Cria o bucket S3 do projeto e sobe a base bruta.
# Uso: bash infra/aws_setup.sh NOME-DO-BUCKET [regiao]
set -euo pipefail

BUCKET="${1:?Informe o nome do bucket (único no mundo, ex.: churn-julio-2026)}"
REGIAO="${2:-us-east-1}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"

if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "Bucket $BUCKET já existe"
elif [ "$REGIAO" = "us-east-1" ]; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGIAO"
else
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGIAO" \
    --create-bucket-configuration LocationConstraint="$REGIAO"
fi

# Bucket privado e versionado (dá para recuperar arquivos sobrescritos).
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled

aws s3 cp "$RAIZ/data/telco_customer_churn.csv" "s3://$BUCKET/raw/telco_customer_churn.csv"

echo
echo "Pronto: s3://$BUCKET/raw/telco_customer_churn.csv"
echo "Próximo passo: criar a external location no Databricks (README, passo 3)."
