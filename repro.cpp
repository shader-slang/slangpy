
#include <slang.h>
#include <slang-com-ptr.h>

template<typename T>
using ComPtr = Slang::ComPtr<T>;

int main()
{
    ComPtr<slang::IGlobalSession> global_session;
    SLANG_RETURN_ON_FAIL(slang::createGlobalSession(global_session.writeRef()));

    ComPtr<slang::ISession> session;
    SLANG_RETURN_ON_FAIL(global_session->createSession({}, session.writeRef()));


    return 0;
}
