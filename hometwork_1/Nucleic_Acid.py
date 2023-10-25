good_letters = {'a', 't', 'g', 'c', 'u'}
complement_dict_DNA = {'a': 't', 'A': 'T', 't': 'a', 'T': 'A', 'g': 'c', 'G': 'C', 'c': 'g', 'C': 'G'}
complement_dict_RNA = {'g': 'c', 'G': 'C', 'c': 'g', 'C': 'G', 'a': 'u', 'A': 'U', 'u': 'a', 'U': 'A'}
transcribe_dict = {'t': 'u', 'T': 'U', 'a': 'a', 'A': 'A', 'g': 'g', 'G': 'G', 'c': 'c', 'C': 'C'}
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
    while True:
        nucl_acid = input('Type the sequence:')
        if not set(nucl_acid.lower()) <= good_letters or {'u', 't'} <= set(nucl_acid.lower()):
            print('No such nucleic acid. Try again!')
        else:
            break

    if command == 'transcribe':
        if 'u' in nucl_acid.lower():
            print('Cannot transcribe RNA. Try again!')
            continue
        new_sequence = ''.join([transcribe_dict[nucl] for nucl in nucl_acid])


    elif command == 'reverse':
        new_sequence = nucl_acid[::-1]

    elif command == 'complement':
        if 'u' in nucl_acid.lower():
            for i in nucl_acid:
                new_sequence += complement_dict_RNA[i]
        else:
            for i in nucl_acid:
                new_sequence += complement_dict_DNA[i]

    elif command == 'reverse complement':
        if 'u' in nucl_acid.lower():
            for i in nucl_acid:
                new_sequence += complement_dict_RNA[i]
        else:
            for i in nucl_acid:
                new_sequence += complement_dict_DNA[i]
        new_sequence = new_sequence[::-1]

    print(new_sequence)




