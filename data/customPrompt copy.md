Job Description (requirements):

{requirements}

Table Headers (column_headers):

{column_headers}

Table Example Rows (example_rows):

{example_rows}


Context:
You are provided with a job description requirements , table headers column_headers, and a few example rows from the table example_rows.

Goal:
Extract the relevant details from the job description to each field in the column_headers. To correctly do this, you must analyze the examples given in example_rows first.

Instructions:

Extract Fields: Use column_headers as a guide to identify which details from requirements match each field. Refer to example_rows to get a better understanding.

Extract Information: For each field in column_headers, extract the most relevant information from requirements.

Leave Empty if Not Found: If the job description does not specify information for a field, leave it blank. Avoid assumptions.

Return your answer in points format without any explanation.

Example:
If column_headers has a field like “Qualifications,” locate relevant qualifications in requirements and map them accordingly.