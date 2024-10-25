provider "aws" {
  region = "us-west-2" # Specify your AWS region
}

resource "aws_security_group" "postgres_sg" {
  name        = "allow_postgres"
  description = "Allow PostgreSQL inbound traffic"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Be cautious with this in production
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "postgres" {
  allocated_storage    = 20
  storage_type         = "gp2"
  engine               = "postgres"
  engine_version       = "16.1" # Ensure the version supports pgvector
  instance_class       = "db.t4g.micro"
  db_name              = var.db_name
  username             = var.db_username
  password             = var.db_password
  vpc_security_group_ids = [aws_security_group.postgres_sg.id]
  skip_final_snapshot  = true
  publicly_accessible  = true
}

resource "null_resource" "db_init" {
  depends_on = [aws_db_instance.postgres]

  provisioner "local-exec" {
    command = <<EOT
      sleep 60  # Wait for 60 seconds to ensure the RDS instance is ready
      export PGPASSWORD=${var.db_password}
      psql -h ${aws_db_instance.postgres.address} -U ${var.db_username} -d ${var.db_name} -c "CREATE EXTENSION IF NOT EXISTS vector;"
    #  psql -h ${aws_db_instance.postgres.address} -U ${var.db_username} -d ${var.db_name} -c "ALTER DATABASE ${var.db_name} SET vector.dimension = 768;"
    EOT
  }
}

output "db_endpoint" {
  value = aws_db_instance.postgres.address
}