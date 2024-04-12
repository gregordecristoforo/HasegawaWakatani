# Vscode tips
Alt + j followed by Alt + o opens the julia REPL
Ctrl + Space - forces autocomplete (In vscode)
Shift+Enter - Run a cell denoted by ##

# Julia REPL
* "]" opens pkg "package manager"
* ";" opens bash
* "?" opens the helper

# Julia tips
* ! marks functions that alters the arguments/parameters/inputs
* && and || are both short circuited - meaning that if the first argument is false in the case of && the second argument is not evaluated, which may be useful in cases where the second argument is dependent on the first being true.
* Bools are numbers
* "*" concatenates strings
* $(Variable) in a string will concatinate it with the rest of the string
* arguments after ";" in function arguments are keywords