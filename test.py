import yaml
import sys



with open('observersInfos.yaml', 'r') as file:
    try:
        observersInfos_str = file.read()
        observersInfos_yamlData = yaml.safe_load(observersInfos_str)
    except yaml.YAMLError as exc:
        print(exc)





new_pattern = ['MocapAligner']


def get_observersInfos():
    # Iterate over the observers
    for observer in observersInfos_yamlData['observers']:
        for body in observer['kinematics']:
            print("body")
            print(body)
            for kine in observer['kinematics'][body]:
                print("kine")
                print(kine)
                for axis in observer['kinematics'][body][kine]:
                    new_pattern.append(axis)

get_observersInfos()

# print(partial_pattern)
print(new_pattern)