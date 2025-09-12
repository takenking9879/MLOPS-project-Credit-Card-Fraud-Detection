output "ecr_uri" {
description = "URI del repositorio ECR"
value = aws_ecr_repository.repo.repository_url
}


output "instance_id" {
description = "ID de la instancia EC2 lanzada"
value = aws_instance.app.id
}


output "public_ip" {
description = "IP pública de la instancia"
value = aws_instance.app.public_ip
}


output "private_key_path" {
description = "Ruta local del PEM generado"
value = local_file.private_key.filename
}