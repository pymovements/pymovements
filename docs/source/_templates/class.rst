{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in methods %}
      ~{{ objname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}{% endblock %}

   {% set properties = property_members.get(module ~ '.' ~ objname, []) %}
   {% if properties %}
   .. rubric:: {{ _('Properties') }}

   .. autosummary::
      :toctree:
      :template: property.rst
   {% for item in attributes if item in properties %}
      ~{{ objname }}.{{ item }}
   {%- endfor %}
   {% endif %}
