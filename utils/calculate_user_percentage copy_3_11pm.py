import pandas as pd

# def calculate_user_points(table, dict_obj):
#     headers = table[0]
#     rows = table[1:]
    
#     # Initialize dictionaries to track points for each individual
#     points = {'Required': {}, 'Preferred': {}}
    
#     def calculate_points(name, row, criteria, point_type):
#         if name not in points[point_type]:
#             points[point_type][name] = 0
        
#         for header, values in criteria.items():
#             if header in headers:
#                 index = headers.index(header)
                
#                 # Ensure the index does not exceed the length of the row
#                 if index >= len(row):
#                     print(f"Skipping row due to missing column data: {row}")
#                     continue
                
#                 cell_value = row[index]
                
#                 if header == 'Experience':
#                     # Check if the 'Experience' field is not null or empty
#                     if cell_value and cell_value.strip():
#                         points[point_type][name] += 1
#                 else:
#                     if isinstance(values, list):
#                         # Check if cell_value is not None and is a string
#                         if cell_value and isinstance(cell_value, str):
#                             cell_values = [value.strip() for value in cell_value.split(',')]
#                             for value in values:
#                                 if value in cell_values:
#                                     points[point_type][name] += cell_values.count(value)
#                     else:
#                         if isinstance(values, list) and values:
#                             if cell_value and isinstance(cell_value, str) and cell_value.strip() == values[0].strip():
#                                 points[point_type][name] += 1
#                         elif not isinstance(values, list):
#                             if cell_value and isinstance(cell_value, str) and cell_value.strip() == str(values).strip():
#                                 points[point_type][name] += 1

#     # Calculate points for each row
#     for row in rows:
#         name = row[0]
#         # Calculate required points
#         calculate_points(name, row, dict_obj.get('Required', {}), 'Required')
#         # Calculate preferred points
#         calculate_points(name, row, dict_obj.get('Preferred', {}), 'Preferred')
    
#     # Prepare the final output
#     result = []
#     for name in points['Required']:
#         result.append({
#             'Name': name,
#             'Required Points': points['Required'].get(name, 0),
#             'Preferred Points': points['Preferred'].get(name, 0)
#         })

#     return result

def calculate_user_points(table, dict_obj):
    headers = table[0]
    rows = table[1:]
    
    # Initialize dictionaries to track points for each individual
    points = {'Required': {}, 'Preferred': {}}
    
    def calculate_points(name, row, criteria, point_type):
        if name not in points[point_type]:
            points[point_type][name] = 0
        
        for header, values in criteria.items():
            if header in headers:
                index = headers.index(header)
                
                # Ensure the index does not exceed the length of the row
                if index >= len(row):
                    print(f"Skipping row due to missing column data: {row}")
                    continue
                
                cell_value = row[index]
                
                if header == 'Experience':
                    # Always give 1 point for the Experience field if it's not empty
                    if cell_value and cell_value.strip():
                        points[point_type][name] += 1
                else:
                    if isinstance(values, list):
                        # Check if cell_value is not None and is a string
                        if cell_value and isinstance(cell_value, str):
                            cell_values = [value.strip() for value in cell_value.split(',')]
                            for value in values:
                                if value in cell_values:
                                    # Increment points based on the number of matches in the array
                                    points[point_type][name] += cell_values.count(value)
                    else:
                        # Single value matching
                        if isinstance(values, list) and values:
                            if cell_value and isinstance(cell_value, str) and cell_value.strip() == values[0].strip():
                                points[point_type][name] += 1
                        elif not isinstance(values, list):
                            if cell_value and isinstance(cell_value, str) and cell_value.strip() == str(values).strip():
                                points[point_type][name] += 1

    # Calculate points for each row
    for row in rows:
        name = row[0]
        # Calculate required points
        calculate_points(name, row, dict_obj.get('Required', {}), 'Required')
        # Calculate preferred points
        calculate_points(name, row, dict_obj.get('Preferred', {}), 'Preferred')
    
    # Prepare the final output
    result = []
    for name in points['Required']:
        result.append({
            'Name': name,
            'Required Points': points['Required'].get(name, 0),
            'Preferred Points': points['Preferred'].get(name, 0)
        })

    return result


# def calculate_points(dict_obj):
#     total_required_points = 0
#     total_preferred_points = 0

#     # Helper function to calculate points dynamically based on type
#     def calculate_dynamic_points(value):
#         if isinstance(value, list):
#             return len(value)
#         elif isinstance(value, str):
#             return 1 if value else 0
#         elif isinstance(value, (int, float)):
#             return 1 if value > 0 else 0
#         else:
#             return 0

#     # Calculate points for the preferred section
#     if 'Preferred' in dict_obj:
#         for key, value in dict_obj['Preferred'].items():
#             print("preferred key :",key,"preferred value :",value)
#             total_preferred_points += calculate_dynamic_points(value)

#     # Calculate points for the required section
#     if 'Required' in dict_obj:
#         for key, value in dict_obj['Required'].items():
#             print("required key :",key,"required value :",value)
#             total_required_points += calculate_dynamic_points(value)

#     return {
#         'total_required_points': total_required_points,
#         'total_preferred_points': total_preferred_points
#     }

def calculate_points(dict_obj):
    total_required_points = 0
    total_preferred_points = 0

    # Helper function to calculate points dynamically based on type
    def calculate_dynamic_points(key, value):
        # Experience field is always worth 1 point
        if key == 'Experience':
            return 1
        # Other fields count the length of the array
        if isinstance(value, list):
            return len(value)
        elif isinstance(value, str):
            return 1 if value else 0
        elif isinstance(value, (int, float)):
            return 1 if value > 0 else 0
        else:
            return 0

    # Calculate points for the preferred section
    if 'Preferred' in dict_obj:
        for key, value in dict_obj['Preferred'].items():
            print("preferred key:", key, "preferred value:", value)
            total_preferred_points += calculate_dynamic_points(key, value)

    # Calculate points for the required section
    if 'Required' in dict_obj:
        for key, value in dict_obj['Required'].items():
            print("required key:", key, "required value:", value)
            total_required_points += calculate_dynamic_points(key, value)

    return {
        'total_required_points': total_required_points,
        'total_preferred_points': total_preferred_points
    }

def calculate_capped_percentage(total_required_and_preferred, users_points):
    total_points = total_required_and_preferred['total_required_points'] + total_required_and_preferred['total_preferred_points']
    total_required_points = total_required_and_preferred['total_required_points']
    
    for user in users_points:
        required_points = user['Required Points']
        preferred_points = user['Preferred Points']
        
        if required_points == total_required_points:
            user_total_points = required_points + preferred_points
            
            # Calculate the percentage based on the comparison, round it off, and format as a string with % sign
            percentage = round((user_total_points / total_points) * 100)
            user['Match %'] = f"{min(percentage, 100)}%"
        else:
            user['Match %'] = "0%"

    return users_points

def insert_percentage_to_table(table, result):
    # Convert the table to a DataFrame
    df = pd.DataFrame(table[1:], columns=table[0])
    
    # Determine the correct key (replace 'Percentage' with the actual key if different)
    percentage_key = 'Match %'  # Replace this with the correct key if needed
    
    # Create a dictionary to map names to percentages, using the correct key
    name_to_percentage = {item['Name']: item.get(percentage_key, 0) for item in result}
    
    # Add the 'Match %' column to the DataFrame
    df['Match %'] = df['Name'].map(name_to_percentage)
    
    # Convert the DataFrame back to a list of lists
    updated_table = [table[0] + ['Match %']] + df.values.tolist()
    
    return updated_table
    
def calculate_user_percentage(table, dict_obj):
    total_required_and_prefered_points = calculate_points(dict_obj)
    print("total_required_and_prefered_points", total_required_and_prefered_points)
    users_required_and_prefered_points = calculate_user_points(table, dict_obj)
    print("users_required_and_prefered_points", users_required_and_prefered_points)
    user_with_percentage = calculate_capped_percentage(total_required_and_prefered_points, users_required_and_prefered_points)
    print("user_with_percentage", user_with_percentage)
    table_with_percentage = insert_percentage_to_table(table, user_with_percentage)
    print("table_with_percentage", table_with_percentage)
    return table_with_percentage
