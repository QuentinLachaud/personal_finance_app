~Readme for finance_app~

Tab activation/deactivation 

**Annoyingly, to add/remove a tab, 3 things need to happen:**

 - Dictionary "display_tab" needs a Bool to determine tab active/inactive state
 - tab list must be expressedly listed to show the tab at all (text only)
 - The tab in question must be listed in the if chain below. Luckily, inactive tabs do not break this if present here.

This is necessary due to how streamlit creates tabs. If instead a tab could be separately created (e.g. st.tab(['example_tab'])) the above would be much more dynamic.
As stated on their website, this is not yet implemented: [streamlit tabs docu](https://docs.streamlit.io/develop/api-reference/layout/st.tabs)

