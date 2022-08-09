# minus_random_num_10_half

transform_method_list = ['concat', 'minus']
ref_data_type_list = ['random', 'pos', 'neg']
ref_num_type_list = ['num', 'IR']
ref_times_dict = {
    'num' : ['10', '20', '30', '40'],
    'IR' : ['1', '2', '3', '4']
}
boundary_type_list = ['half', '1', '3']

for transform_method in transform_method_list:
    for ref_data_type in ref_data_type_list:
        for ref_num_type in ref_num_type_list:
            cur_time_list = ref_times_dict[ref_num_type]
            for ref_times in cur_time_list:
                for boundary_type in boundary_type_list:
                    print('{0}_{1}_{2}_{3}_{4}'.format(transform_method, ref_data_type, ref_num_type, ref_times, boundary_type))