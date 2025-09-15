+------------------------AWS VPC ----------------------------+
|                                                            |
|  +-------------------+      +--------------------------+   |
|  |   Public Subnet   |      |      Private Subnet      |   |
|  |-------------------|      |--------------------------|   |
|  |                   |      |                          |   |
|  |   EC2 Instance    |      |  +------------------+    |   |
|  | (for mgmt/NAT)    |      |  |   **RDS/Aurora** |    |   |
|  |                   |      |  |    movie DB +    |    |   |
|  +-------------------+      |  |    vector DB     |    |   |
|        ^       ^            |  +------------------+    |   |
|        | Embed |            |            ^             |   |
|        | RAG   |            |            |             |   |
|        | API   +----------------> **Lambda Function**  |   |
|        |                    |            |             |   |
|        |                    +------------|-------------+   |
|        |                                 |                 |
+--------|---------------------------------|-----------------+
         |                                 |                    
         v                                 v                           
+-------------------+              +-------------------------+    
| Nomic/ Gemini API |              | REST API calling Lambda |          
+-------------------+              +-------------------------+  
                                                 |
                                       +-------------------+
                                       |   Frontend (S3    |
                                       |   or web client)  |
                                       +-------------------+





Lambda Function:
  - Receives REST API requests from API Gateway
  - Calls external LLM & embedding APIs (outside AWS, via NAT/IGW)
  - Queries RDS (movie.db, vector DB)
  - Returns results to frontend via API Gateway