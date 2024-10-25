variable "db_username" {
  description = "The username for the PostgreSQL database"
  type        = string
  default     = "admin"
}

variable "db_password" {
  description = "The password for the PostgreSQL database"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "The name of the PostgreSQL database"
  type        = string
  default     = "mydatabase"
}

variable "vpc_id" {
  description = "The VPC ID"
  type        = string
}
