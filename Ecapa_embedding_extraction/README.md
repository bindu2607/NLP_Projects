from graphviz import Digraph

# Create a new directed graph for the UML Class Diagram
dot = Digraph(comment='UML Class Diagram - Student, Exam, Subject', format='png')

# Common node style
node_style = {
    'shape': 'record',
    'fontname': 'Helvetica',
    'fontsize': '10'
}

# Add Student class
dot.node('Student',
         '''{Student|
         - studentID : int\\l
         - name : string\\l
         - course : string\\l|
         + registerExam()\\l
         + viewResults()\\l}''',
         **node_style)

# Add Exam class
dot.node('Exam',
         '''{Exam|
         - examID : int\\l
         - date : date\\l
         - type : string\\l|
         + conductExam()\\l
         + evaluateExam()\\l}''',
         **node_style)

# Add Subject class
dot.node('Subject',
         '''{Subject|
         - subjectCode : string\\l
         - subjectName : string\\l
         - credits : int\\l|
         + addSubject()\\l
         + removeSubject()\\l}''',
         **node_style)

# Add relationships
# Student <-> Exam (many-to-many)
dot.edge('Student', 'Exam', label='*', arrowhead='none', taillabel='*')

# Exam -> Subject (many-to-one)
dot.edge('Exam', 'Subject', label='1', arrowhead='none', taillabel='*')

# Save and render diagram
output_path = '/mnt/data/student_exam_subject_class_diagram'
dot.render(output_path, cleanup=True)

output_path + '.png'
