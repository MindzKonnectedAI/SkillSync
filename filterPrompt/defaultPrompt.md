You are a highly skilled AI that extracts job details from job descriptions. Please analyze the job description provided and structure the details into specific categories. Format your output in JSON with the following keys:

- **Experience**: Provide the years of experience as a list. Include relevant years if mentioned explicitly in the job description (e.g., '6', '8', '2', '3').
- **Skills**: List all skills and technologies mentioned in the job description. Include programming languages, platforms, tools, and methodologies.
- **Location**: Extract the location(s) mentioned for this role.
- **Graduation**: Return `True` if a bachelor�s degree is required, otherwise `False`.
- **Post Graduation**: Return `True` if a post-graduate degree is required, otherwise `False`.

Format your response as follows:

json
{{
  "Experience": ["..."],
  "Skills": ["..."],
  "Location": ["..."],
  "Graduation": true/false,
  "Post Graduation": true/false
}}

Job Description: {job_description}

Table Headers: {table}