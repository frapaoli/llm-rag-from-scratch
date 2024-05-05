import src.constants as constants

def menu():

    # Prompt the user menu request until the user selection is valid, then return it
    while True:
        menu_selection = input(constants.USER_MENU_REQUEST)
        print("\n")

        if menu_selection in constants.MENU_SELECTION_TO_USER_OPTION_MAPPING.keys():
            return constants.MENU_SELECTION_TO_USER_OPTION_MAPPING[menu_selection]
        
        print("Invalid option. Please try again.\n")
