good_letters = ['a', 't', 'g', 'c', 'u']
complement_dict_DNA = {'a': 't', 'A': 'T', 't': 'a', 'T': 'A', 'g': 'c', 'G': 'C', 'c': 'g', 'C': 'G'}
complement_dict_RNA = {'g': 'c', 'G': 'C', 'c': 'g', 'C': 'G', 'a': 'u', 'A': 'U', 'u': 'a', 'U': 'A'}
known_commands = ['exit', 'transcribe', 'reverse', 'complement', 'reverse complement']
while True:
    new_sequence = ''
    command = input('Type the command:')
    if command == 'exit':
        print('Good luck!')
        break

    elif command not in known_commands:
        print("Don't know this command. Try again!" )
        continue

# If the command is in the list and is not 'exit' - we should read the sequence first.
    condition = 'no such acid'
    while condition == 'no such acid':
        nucl_acid = input('Type the sequence:')
        condition = 'stop'
        for i in nucl_acid.lower():
            if i not in good_letters:
                condition = 'no such acid'
        if 'u' in nucl_acid.lower() and 't' in nucl_acid.lower():
            condition = 'no such acid'
        if condition == 'no such acid':
            print('No such nucleic acid. Try again!')

    if command == 'transcribe':
        if 'u' in nucl_acid.lower():
            print('Cannot transcribe RNA. Try again!')
        else:
            for i in nucl_acid:
                if i == 't':
                    new_sequence += 'u'
                elif i == 'T':
                    new_sequence += 'U'
                else:
                    new_sequence += i
        print(new_sequence)

    elif command == 'reverse':
        new_sequence = nucl_acid[::-1]
        print(new_sequence)

    elif command == 'complement':
        if 'u' in nucl_acid.lower():
            for i in nucl_acid:
                new_sequence += complement_dict_RNA[i]
        else:
            for i in nucl_acid:
                new_sequence += complement_dict_DNA[i]
        print(new_sequence)

    elif command == 'reverse complement':
        if 'u' in nucl_acid.lower():
            for i in nucl_acid:
                new_sequence += complement_dict_RNA[i]
        else:
            for i in nucl_acid:
                new_sequence += complement_dict_DNA[i]
        new_sequence = new_sequence[::-1]
        print(new_sequence)




