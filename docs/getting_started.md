# Getting started with PyAstroDyn-edu

Step-by-step instructions for Windows, macOS and Linux: install the tools,
install the library, read the code, run a first example and a first
notebook, and learn with an example problem. Plan for about 30 minutes for
Steps 1 to 10. No previous experience with Python is
assumed.

You will install three things:

| What | Why |
|---|---|
| **Python** (version 3.11 or newer) | The language the library is written in |
| **Visual Studio Code** ("VS Code") | A free editor, to read the code and to run scripts and notebooks |
| **PyAstroDyn-edu** | The library, with its examples and notebooks |

Note: *Visual Studio Code* is not the same program as *Visual Studio*. You
need Visual Studio Code.

Throughout this guide, the **Command Palette** of VS Code is opened with
`Ctrl+Shift+P` on Windows and Linux, and with `Cmd+Shift+P` (⇧⌘P) on macOS.
You then type the first letters of a command and select it from the list.

**Commands for the terminal** are shown in grey boxes. A command is a line of
words separated by spaces: `python -m pip` is three words. Copy the line and
paste it into the terminal, or type it exactly, spaces included, and press
Enter. What each word means is explained where the command first appears.

---

## Step 1. Install Python

First check whether you already have it. Open a terminal (Windows: the
*Terminal* or *PowerShell* app; macOS: the *Terminal* app; Linux: your
terminal) and type:

| System | Command |
|---|---|
| Windows | `python --version` |
| macOS, Linux | `python3 --version` |

This command only prints the version of the Python on your computer; it
changes nothing. If the answer is `Python 3.11` or a higher number, go to
Step 2. Otherwise:

**Windows.** Install the *Python install manager*, either from the Microsoft
Store (search for "Python install manager") or from
<https://www.python.org/downloads/>. Then close the terminal, open a new one,
and type `python --version` again. The first time, it may offer to download
Python: accept.

**macOS.** Download the macOS installer from
<https://www.python.org/downloads/> and run it. Then open a new terminal and
type `python3 --version`.

**Linux.** Python is normally already installed. On Ubuntu and Debian, make
sure the modules for virtual environments are there:

```
sudo apt install python3 python3-venv python3-pip
```

Here `sudo` means "as administrator" (it asks for your password), `apt` is
the program that installs software on Ubuntu and Debian, and the three names
are the packages to install. On other distributions use the corresponding
packages of your package manager.

## Step 2. Install Visual Studio Code and two extensions

1. Download VS Code from <https://code.visualstudio.com/> and install it.
2. Start VS Code. In the bar on the left, open the **Extensions** view (the
   icon with four squares).
3. Install the two extensions of the table below: type the identifier in the
   search box, select the extension, and press **Install**.

| Extension | Publisher | Identifier | What it is for |
|---|---|---|---|
| **Python** | Microsoft | `ms-python.python` | Connects VS Code to your Python |
| **Jupyter** | Microsoft | `ms-toolsai.jupyter` | Opens and runs notebooks inside VS Code |

Several extensions have similar names. The identifier is unique: check it on
the page of the extension before you install. A few helper extensions
(for example Pylance, Jupyter Keymap, Jupyter Notebook Renderers) are
installed automatically with these two; that is expected.

**Optional: JupyterLab, to run notebooks in a web browser.** JupyterLab is
*not* an extension of VS Code and is not in the Extensions view. It is a
separate program that runs on your computer and shows the notebooks in a page
of your web browser. It is the same on Windows, macOS and Linux, and it is
installed with one command in Step 6. You need it only if you prefer the
browser to VS Code for the notebooks; see "Alternative: the notebook in a web
browser (JupyterLab)" after Step 11.

## Step 3. Get the library

The library is on GitHub at:

> <https://github.com/pyastrodyn/pyastrodyn-edu>

1. Open that page in a browser.
2. Press the green **Code** button and choose **Download ZIP**.
3. Unzip the file. You get a folder called `pyastrodyn-edu-main`.
4. Move the folder to a place you will find again, for example your
   *Documents* folder. Avoid folders synchronised with a cloud service
   (OneDrive, iCloud, Dropbox): they can interfere with the next steps.

If you know Git, you can instead clone the repository; the rest of the guide
is the same.

## Step 4. Open the folder in VS Code

1. In VS Code choose **File > Open Folder...** and select the folder
   `pyastrodyn-edu-main` (the one that contains the file `README.md`).
2. If VS Code asks "Do you trust the authors of the files in this folder?",
   answer **Yes**.

The **Explorer** on the left now shows the contents:

```
pyastrodyn-edu-main/       the project: the folder you opened
    README.md
    LICENSE
    pyproject.toml
    docs/  examples/  notebooks/  tests/  tools/
    pyastrodyn_edu/        the library itself: the Python code
        elements.py, twobody.py, ...
```

**Two folders with almost the same name: this is normal.** It is the
standard layout of a Python project.

- The outer folder, `pyastrodyn-edu` with a hyphen (plus `-main` when it
  comes from a ZIP download), is the whole project: the library together
  with its documentation, examples, notebooks and tests.
- The inner folder, `pyastrodyn_edu` with an underscore, is the library
  itself. Its name is the one you type in Python: `import pyastrodyn_edu`.
  Python does not allow a hyphen in such a name, hence the underscore.

Always open the outer folder in VS Code, and type the commands of this guide
there.

## Step 5. Create a virtual environment

A virtual environment is a private copy of Python inside the folder. The
library and the packages it needs are installed there, so nothing else on
your computer is changed, and you can delete it at any time.

There are two ways to create it. Way A, in the terminal, works on every
system and with every version of VS Code. Way B uses a command of VS Code
that is quicker, but its name changes between versions and in some versions
it does not appear. If you are not sure, use Way A.

### Way A: in the terminal

1. Open a terminal inside VS Code: **Terminal > New Terminal**. On macOS
   the menus of VS Code, **Terminal** included, are in the bar at the very
   top of the screen. On every system the keys ``Ctrl+Shift+` `` do the same
   (the `Ctrl` key also on macOS; the last key is the backtick). The
   terminal opens as a panel at the bottom of the VS Code window, already
   inside the folder you opened in Step 4.
2. Create the environment:

| System | Command |
|---|---|
| Windows | `python -m venv .venv` |
| macOS, Linux | `python3 -m venv .venv` |

   That is: start Python, run its module `venv` (virtual environment), and
   make the environment in a new folder called `.venv`. It takes a few
   seconds and prints nothing. The folder `.venv` appears in the Explorer.

3. Tell VS Code to use it. VS Code usually shows a message at the bottom
   right, asking whether to use the new environment for this workspace:
   answer **Yes**. If no message appears, open the Command Palette, run
   **Python: Select Interpreter**, and choose the entry that contains
   `.venv`.
4. Close the terminal (the bin icon at the top right of the terminal panel)
   and open a new one with **Terminal > New Terminal**. The line where you
   type now starts with `(.venv)`: the environment is active.

If the line does not start with `(.venv)`, activate the environment by hand
in the terminal:

| System | Command |
|---|---|
| Windows | `.venv\Scripts\activate` |
| macOS, Linux | `source .venv/bin/activate` |

If that also fails, see "If something goes wrong" at the end.

### Way B: with the Command Palette

1. Open the Command Palette and run **Python: Create Environment**. If the
   command is not in the list, type only `Create Environment`; if it is
   still not there, use Way A.
2. Choose **Venv**.
3. Choose the Python version you installed in Step 1 (3.11 or newer).
4. If VS Code asks which dependencies to install, tick **dev** and confirm.
   If it does not ask, never mind: Step 6 does it.
5. Wait for the message that the environment is ready. A new folder `.venv`
   appears in the Explorer.

## Step 6. Install the library PyAstroDyn-edu

1. Use the terminal of Step 5, or open a new one inside VS Code:
   **Terminal > New Terminal** (see Step 5, Way A, point 1).
2. Check that the line where you type starts with `(.venv)`. That means the
   virtual environment is active. If it does not, see "If something goes
   wrong" at the end.
3. Type this command and press Enter:

```
python -m pip install -e ".[dev]"
```

This installs the library together with NumPy, SciPy, Matplotlib and what is
needed for the notebooks and the tests. It takes one or two minutes and ends
with a line that begins `Successfully installed`.

The command, word by word:

| Part | Meaning |
|---|---|
| `python` | Start Python. Because `(.venv)` is active, it is the Python of the virtual environment of Step 5 |
| `-m pip` | "Run the module named `pip`". `pip` is the standard tool of Python that installs packages. Writing `python -m pip`, and not only `pip`, makes sure that the packages go into this environment |
| `install` | What `pip` must do: install |
| `-e` | "Editable": do not copy the library; use the files in your folder directly. If you change a file, the change is active at once |
| `.` | The dot means "the project in the folder where I am": PyAstroDyn-edu. Nothing is downloaded for the library itself |
| `[dev]` | Install also the optional group of packages called `dev`: what the tests, the figures and the notebooks need |
| `"` ... `"` | The quotation marks stop the terminal from giving the square brackets a meaning of its own |

The packages that the library needs (NumPy, SciPy and the others) are
downloaded from the internet by this command.

**Optional (JupyterLab, see Step 2).** To run the notebooks in a web browser
as well, type one more command, the same on Windows, macOS and Linux:

```
python -m pip install jupyterlab
```

It reads like the command above: run `pip`, and install the package named
`jupyterlab`, which is downloaded from the internet.

## Step 7. Check the installation

In the same terminal type:

```
python -m pytest
```

It means "run the module named `pytest`": the testing tool installed in
Step 6. It finds the files in the folder `tests` and runs the tests of the
library, a few hundred, in one or two minutes. The last line must say that they all `passed`, with no `failed`. If
it does, everything works.

## Step 8. Read the code

This step shows how to open and read the files. How to *learn* from them,
one example problem at a time (notebook first, then script, then the code of
the library), is the subject of **Step 11**; Steps 9 and 10 prepare it.

**First, how to read the `.md` files.** The files whose name ends in `.md`
(`README.md` and those in `docs`) are text documents with simple formatting
marks. VS Code can show them formatted, with headings and tables: this is
called the **preview**. No setting is needed.

| To do this | Windows, Linux | macOS |
|---|---|---|
| Show the open file formatted, in a new tab | `Ctrl+Shift+V` | `Cmd+Shift+V` |
| Show the text and the formatted view side by side | `Ctrl+K`, release, then `V` | `Cmd+K`, release, then `V` |
| The same with the mouse | Right-click the file in the Explorer, **Open Preview** | the same |

To make every `.md` file open formatted when you click it:

1. Right-click any `.md` file in the Explorer and choose **Open With...**
2. In the list that appears, choose **Configure default editor for
   '*.md'...**
3. Choose **Markdown Preview**.

To edit a `.md` file afterwards, right-click it and choose **Open With...**,
then **Text Editor**.

**Then, what to read.** Click a file in the Explorer to open it. A good order
for a first reading:

| Order | File | What it is |
|---|---|---|
| 1 | `README.md` | Purpose, conventions, units |
| 2 | `pyastrodyn_edu/single_state.py` | The three hardest algorithms, written for one state with ordinary `if` statements |
| 3 | `docs/numpy_primer.md` | The six NumPy idioms that the rest of the code uses |
| 4 | `pyastrodyn_edu/elements.py` | The same algorithms for many states at once |
| 5 | `docs/problems.md` | Worked problems with data and results |
| any time | `docs/functions.md` | The function guide: a map of the library and every function explained |

Two things that make reading the code easier:

- **What does a function do?** Rest the mouse on its name anywhere in the
  code: its documentation appears.
- **Where is a function written?** Click its name and press `F12`
  (**Go to Definition**).

Every function follows the same pattern: a description, the inputs and
outputs with their shapes, the books and the lecture it comes from, and then
the code in numbered steps.

## Step 9. Run a first example

1. In the Explorer open `examples/first_steps.py`.
2. Press the **Run** button, the triangle at the top right of the editor
   (**Run Python File**).

The terminal shows three results: a position and velocity, an orbital period
of about 153.6 minutes, and the position a quarter of a period later. The
last line confirms that the library works.

You can also run it by typing in the terminal:

```
python examples/first_steps.py
```

That is: start Python and run the file `first_steps.py` in the folder
`examples`.

Now change something, for example the eccentricity `0.15` in part 1, save
the file, and run it again.

## Step 10. Run a notebook in VS Code

A notebook is a file whose name ends in `.ipynb`. It mixes text, code and
figures. The code is in **cells** (boxes) that you run one at a time; the
results of a cell appear directly under it.

**The kernel.** A notebook does not run by itself: it needs a Python to do
the computing, called its **kernel**. You must tell each notebook once which
Python to use, and it must be the one of the virtual environment of Step 5,
`.venv`, because that is where the library is installed. The current choice
is shown at the top right of the notebook:

| What the top right shows | Meaning | What to do |
|---|---|---|
| **Select Kernel** | No kernel chosen yet; nothing can run | Choose one, as in point 2 below |
| `.venv (Python 3.x.x)` | The right kernel | Nothing |
| Another Python, without `.venv` | A Python that does not have the library | Click the name and choose again, as in point 2 below |

**Run the notebook.**

1. In the Explorer open `notebooks/Lecture01_numerical_integration.ipynb`.
2. At the top right of the notebook press **Select Kernel**. A list opens at
   the top centre of the window: choose **Python Environments...**, and then
   the entry that contains `.venv`.
3. Press **Run All** at the top of the notebook. The cells run from top to
   bottom; numbers and figures appear under each one. The first time, VS
   Code may ask to choose a kernel or to install `ipykernel`: accept, and
   nothing runs until you have answered.
4. To run a single cell, click in it and press `Shift+Enter`.

**How to see that a cell has run.** Look at the square brackets at the left
of a code cell:

| At the left of the cell | Meaning |
|---|---|
| `[ ]` (empty) | The cell has not run |
| A turning circle, or a clock | The cell is running, or waiting for the cells before it |
| `[3]` with a green tick | The cell ran, as the third one; its results are under it |
| `[3]` with a red cross | The cell stopped with an error, shown under it |

A cell of text has no brackets, and some code cells print nothing (for
example a cell that only contains `import` lines): scroll down to the first
cell that prints.

Change a number in a cell, for example the eccentricity of the test orbit,
press **Run All** again, and see the effect.

In a notebook cell you can ask for the documentation of any function by
typing its name followed by a question mark, for example `ad.coe2rv?`.

## Step 11. Learn with an example problem

The library is learned one **example problem** at a time. Each example
problem belongs to a lecture and exists in two forms with the same content:
a notebook, to experiment, and a script, to read.

The example problems are in the folders `examples/lecture01`,
`examples/lecture02` and so on, one folder per lecture. The file `examples/README.md`
lists them; their statements and results are also in `docs/problems.md`.
The file names begin with the number of the problem: Example problem 4.1 is
`example_4_1_...`.

The parts below use **Example problem 4.1, Where is the spacecraft 40
minutes after perigee?** It is a good first one; afterwards take the
examples of the lecture you are studying, in order.

For each example problem, follow these six parts.

**11.1 Read the problem.** In the Explorer open
`examples/lecture04/example_4_1_position_at_a_given_time.ipynb`. The first cell
states the problem. Before running anything, estimate the answer, or solve
the problem by hand with the equations of the lecture.

**11.2 Experiment in the notebook.**

1. Select the kernel as in Step 10, point 2: **Select Kernel**, **Python
   Environments...**, the entry that contains `.venv`. The top right of the
   notebook must then show `.venv`.
2. Press **Run All**. Results and the figure appear under the cells.
3. Change a number in section 1, "Given data", for example
   `t = 3600.0`, and press **Run All** again. Compare the results
   with what you expected.
4. To return to the original problem, type the original value back (it is
   written in the comment at the end of the line) and press **Run All**; the
   last section then prints "The results agree with the stored results". To
   discard all your changes of the notebook at once, run
   `python tools/make_examples.py` in the terminal and reopen the notebook.

**11.3 Read the script.** Open
`examples/lecture04/example_4_1_position_at_a_given_time.py`: the same solution as
ordinary Python, to read from top to bottom.

- Lines that begin with `#` are text; the others are code.
- Press the **Run** triangle at the top right, as in Step 9, to run the
  whole script: the results are printed in the terminal.
- To go slowly, use the debugger: click to the left of a line number to put
  a red dot (a breakpoint) there, and choose **Run > Start Debugging**
  (`F5`); if a list appears, choose **Python Debugger** and then **Python
  File**. The program stops at the dot. `F10` runs one line; the values of
  all variables are shown on the left. `Shift+F5` stops.

**11.4 Go from the script into the library.** In the script, click the name
of a library function, for example `M2E` in `ad.M2E(...)`, and press
`F12`. VS Code opens the file of the library where the function is written.
Read first its description (inputs, outputs, books and lecture, limits of
validity), then its code, which follows numbered steps. While debugging,
`F11` on a line that calls a library function steps into it.
`docs/functions.md` lists all the functions, module by module.

**11.5 Check and vary.** The last section of the example compares the
results with the stored results of the problem; this comparison holds only
for the original data. Then do the "Now try" questions at the end.

Where the example problems come from: their statements, data and stored
results are in one place in the library, and the automatic tests solve every
problem again at each change of the code and compare. This is how the
library is validated. An example problem shows a complete solution; it is a
worked example, not an exercise.

**11.6 Make it your own.** The script is ordinary Python: use it as the
starting point of your own work (other orbits, a loop over many times or
eccentricities, more figures). Work on a copy, not on the original:

1. In the Explorer, make a new folder called `my_work` in the project
   (right-click an empty area, **New Folder...**).
2. Right-click the script, choose **Copy**, then right-click `my_work` and
   choose **Paste**. Rename the copy, for example `my_orbit_study.py`.
3. Change and run the copy freely. The library works from any folder of the
   project.

Why not change the original:

- The notebook is made from it by `tools/make_examples.py`. After a change
  of the original, the notebook no longer matches, and one of the tests of
  Step 7 fails, until you run `python tools/make_examples.py` in the
  terminal.
- When you install a new version of the library, the original files are
  replaced. Copy your folder `my_work` to the new folder first (see
  "Updating the library later").
- Every file called `example_*.py` in a folder `examples/lectureNN` is run
  by the tests; your own experiments do not belong there.

## Alternative: the notebook in a web browser (JupyterLab)

You do not need this if Step 10 works. JupyterLab is the classical way to
use notebooks: a page in your web browser, outside VS Code. Nothing is sent
to the internet; the page talks to a program on your own computer.

1. Install JupyterLab, if you did not do it in Step 6. In the VS Code
   terminal (with `(.venv)` at the start of the line) type:

```
python -m pip install jupyterlab
```

2. Start it, in the same terminal:

```
python -m jupyter lab
```

   That is: run the module named `jupyter`, and in it the program `lab`.

3. Your web browser opens with JupyterLab. If it does not, copy the address
   that begins with `http://localhost:8888` from the terminal into the
   browser.
4. In the file list on the left, open the folder `notebooks` and
   double-click a notebook.
5. Choose **Run > Run All Cells**.
6. When you have finished, close the browser tab, go back to the terminal and
   press `Ctrl+C` to stop JupyterLab (the `Ctrl` key also on macOS).

**Starting JupyterLab without opening VS Code.** Open the terminal of your
system, go to the folder of the library with the command `cd` followed by
the path of the folder, and type:

| System | Command |
|---|---|
| Windows | `.venv\Scripts\python -m jupyter lab` |
| macOS, Linux | `.venv/bin/python -m jupyter lab` |

## Updating the library later

When a new version is published, download the ZIP again and repeat Steps 4 to
7 in the new folder. If you changed files in the old folder and want to keep
them, copy them somewhere else first.

## If something goes wrong

| What you see | What to do |
|---|---|
| `python` is "not recognized" or "not found" in the terminal | Close the terminal and open a new one. On macOS and Linux try `python3`. If it still fails, repeat Step 1. |
| The terminal line does not start with `(.venv)` | Open the Command Palette, run **Python: Select Interpreter**, choose the entry that contains `.venv`, then open a new terminal with **Terminal > New Terminal**. |
| Still no `(.venv)` | Use the environment directly. Windows: `.venv\Scripts\python -m pip install -e ".[dev]"`. macOS and Linux: `.venv/bin/python -m pip install -e ".[dev]"`. Use the same prefix for the other commands. |
| **Python: Create Environment** is not in the Command Palette | Create the environment in the terminal: Step 5, Way A. |
| **Python: Create Environment**, or `python3 -m venv .venv`, fails on Linux | Install the package `python3-venv` (Step 1) and try again. |
| Windows: the activation command of Step 5 gives an error about "scripts" being "disabled" | Skip the activation and use the prefix `.venv\Scripts\python` instead of `python` (see the row "Still no `(.venv)`" above). |
| `pip` cannot install NumPy or SciPy, with a long error about "building" | Your Python is probably newer than the packages support. Install Python 3.13 beside it (Windows: `py install 3.13`), delete the folder `.venv`, and repeat Step 5 choosing that version. |
| `ModuleNotFoundError: No module named 'pyastrodyn_edu'` | The library is not installed in the environment in use. Repeat Step 6, and in a notebook repeat point 2 of Step 10. |
| The notebook asks to install `ipykernel` | Accept; or repeat Step 6, which installs it. |
| **Run All** was pressed but no results appear | Check the top right of the notebook: if it shows **Select Kernel**, no kernel is chosen (Step 10, point 2). Check the brackets at the left of the cells: `[ ]` means not run. Answer any question VS Code shows at the top centre or bottom right. |
| The notebook shows no **Select Kernel** entry with `.venv` | Close the notebook, run **Python: Select Interpreter** and choose `.venv`, then reopen the notebook. |
| Some tests fail in Step 7 | Copy the last twenty lines of the output and send them to the author. |

## What was installed, and how to remove it

- Everything specific to the library is inside its folder: the files you
  downloaded and the environment in `.venv`. To remove it all, delete the
  folder.
- Python and VS Code are ordinary programs and are removed in the usual way
  of your operating system.

---

*The steps of this guide were tested on Linux by running every command. The
Windows and macOS steps follow the documentation of Python and of VS Code;
menu names in VS Code change slightly from one version to the next.*
