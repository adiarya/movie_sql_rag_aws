A RAG (Retrieval Augmented Generation) workflow for answering statistical and natural language queries.

Assume we have the following scenario: We have a movie database (with various tables and info like movies, cast, ratings, genres, plots, runtime etc.). A user asks natural language queries. Our task is to translate the natural language, and retrieve the answer. A simple RAG workflow (storing the info as a vector database, and searching for the embedded user query) works great for questions that only need info from a few movies (for example: "What was the movie about a killer robot sent from the future?", "What was that movie directed by Nolan starring Anne Hathaway?" etc.). However for statistical queries (like "What is the average runtime of action movies rated above 9", or "In what decade does Scorsese have the most movies?"), we might have to look at a large subset of movies, and therefore a similarity search followed by a RAG query might not work. It would be far easier to translate the user query to a SQL based query (via coding LLMS) and display the result. This project handles the complexities of a user query and retieves the result either via a traditional RAG workflow, or a natural language to SQL conversion.

We store the info about the IMDb top 250 movies in database, and show how to achieve the above in a commercial cloud setting (on Amazon Web Services). To embed both the original movie information we use Nomic AI, and for LLM queries use the currently free Gemini API.

This project mostly serves as a demonstration of implementing a decently complicated RAG workflow resulting in a far more complicated cloud setup. The focus was on efficiency in terms of cost, memory, runtime in that order. We use free services wherever possible. In a production environment although the overall architecture, and workflow would remain the same, there are various improvemnents possible.

This is what our architecture and their connections look like on AWS. We use various cost cutting measures like hosting the frontend on S3, using a serverless implementation on Lambda, external API calls rather than Amazon Bedrock Services, an EC2 NAT instance instead of a gateway etc.

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
| Nomic/ Gemini API |              |**REST API**calls Lambda |          
+-------------------+              +-------------------------+  
                                                 |
                                       +-------------------+
                                       |  Frontend **S3**  |
                                       |     User query    |
                                       +-------------------+

Lambda Function:
  - Receives REST API requests from API Gateway
  - Calls external LLM & embedding APIs (outside AWS, via EC2 NAT)
  - Queries RDS (movie.db, vector DB)
  - Returns results to frontend via API Gateway