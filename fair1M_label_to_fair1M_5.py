import os
# from jdet.config.constant import FAIR1M_1_5_CLASSES
# FAIR1M_1_5_CLASSES = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 
#         "Football_Field", "Baseball_Field", 'Intersection', 'Roundabout', 'Bridge'
#     ]

# FAIR_CLASSES_ = ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 
#         'A220', 'A321', 'A330', 'A350', 'ARJ21', 'other-airplane', 'Passenger_Ship', 
#         'Motorboat', 'Fishing_Boat', 'Tugboat', 'Engineering_Ship', 'Liquid_Cargo_Ship', 
#         'Dry_Cargo_Ship', 'Warship', 'other-ship', 'Small_Car', 'Bus', 'Cargo_Truck', 
#         'Dump_Truck', 'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck_Tractor', 
#         'other-vehicle', 'Basketball_Court', 'Tennis_Court', 'Football_Field', 
#         'Baseball_Field', 'Intersection', 'Roundabout', 'Bridge'
#     ]

NeedChange_Airplane = {'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 
        'A220', 'A321', 'A330', 'A350', 'ARJ21', 'other-airplane', }

NeedChange_Ship = {'Passenger_Ship', 'Motorboat', 'Fishing_Boat', 'Tugboat', 'Engineering_Ship', 'Liquid_Cargo_Ship', 
        'Dry_Cargo_Ship', 'Warship', 'other-ship',}

NeedChange_Vehicle = {'Small_Car', 'Bus', 'Cargo_Truck', 'Dump_Truck', 'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck_Tractor', 
        'other-vehicle',}



def fair1m_label_to_fair_1m_5(source_dataset_path_task):

    
    lb_path = os.path.join(source_dataset_path_task, 'labelTxt')
    filelist = os.listdir(lb_path)
    for filename in filelist:
        de_path = os.path.join(lb_path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(".txt"): #Specify to find the txt file.
                reWriteLabel(de_path)
        else:
            print(de_path,':not a label path')



def reWriteLabel(txt_path):
    def return_line(lb, line):
        if lb in NeedChange_Airplane:
            return line.replace(lb, 'Airplane')
        if lb in NeedChange_Ship:
            return line.replace(lb, 'Ship')
        if lb in NeedChange_Vehicle:
            return line.replace(lb, 'Vehicle')      
        return line 

    new_label = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ds = line.split(" ")
            if len(ds) < 10:
                new_label.append(ds)
                continue           
            lb =ds[8]
            new_label.append(return_line(lb))
    
    f = open(txt_path, 'w')
    for nl in new_label:
        f.write(nl+'\n')
    f.close()