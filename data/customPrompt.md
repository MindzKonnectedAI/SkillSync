Given a job description and a set of dynamic table headers, extract details from the job description and map them to the appropriate fields in the table headers.


*Job Description:*
{requirements}

*Table Headers:*
{column_headers}

### Instructions:
1. For each field in the table headers, identify the relevant details from the job description.
2. Map each extracted detail to its corresponding header and Sepcified required and desired criteria on the basis of boolean string provided in job descrption. Always marks "required" criteria explicitly as "required" and "desired" criteria explicitly as "desired," and never forgets to do so in the format below:

    - *{column_headers}:* [Extracted detail, e.g., string for text fields, numeric for experience, etc.]

3. If any detail is unavailable or does not match a header, return "Not Available."
4. If any field in the table headers is not specified in the job description or does not have in corresponding details in the job description, never metion in output.
5. Sepcified required and desired criteria on the basis of boolean string provided in job descrption. Always marks "required" criteria explicitly as "required" and "desired" criteria explicitly as "desired," and never forgets to do so.
6. Don't give any explanation for your answer. Just return the answer.

### Example Output:
If the table headers include fields such as "Name," "Location," "Experience," and "Skills," the output should look like this:

- *Name:* Solution Architect
- *Location:* Gurugram, India
- *Experience:* [Extract or approximate years if available]
- *Skills:* .NET Core, Microservices, Cloud, Azure, etc.
- *Graduation:* Bachelor's in Computer Science
- *Post Graduation:* Master's (if applicable)
- *PhD:* Not Available

